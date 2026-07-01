#pragma once

// =============================================================================
// unitree_mujoco 侧 — 特权信息话题发布器 (DDS Channel Publisher)
//
// 话题名称: rt/privileged_state (DDS, 与 LowState 同一条总线)
// 消息体:   PrivilegedStateMsg (128 bytes, 固定大小)
//
// 集成方式: 在 G1Bridge::run() 中调用 publish(), 由 DDS bridge 线程驱动
// =============================================================================

#include <mujoco/mujoco.h>

#include <unitree/robot/channel/channel_publisher.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>


// =============================================================================
// PrivilegedStateMsg — 128 bytes, 与 unitree_rl_lab 侧定义严格一致
//
// 如需 IDL 代码生成, 参考 simulate/idl/PrivilegedState.idl
// =============================================================================

struct PrivilegedStateMsg
{
    float    end_effector_pos_w[15];  // 5 bodies × 3 (60 bytes)
    float    root_pos_w[3];           // root link world pos (12 bytes)
    float    box_pos_w[3];            // box world pos (12 bytes)
    float    box_quat_w[4];           // box quat wxyz (16 bytes)
    float    box_size[3];             // box half-size [w,d,h] (12 bytes)
    float    goal_pos_w[3];           // goal world pos (12 bytes)
    uint32_t sequence;                // frame counter (4 bytes)
    uint32_t padding;                 // alignment (4 bytes)
};

static_assert(sizeof(PrivilegedStateMsg) == 128, "Msg size must be 128 bytes");


// =============================================================================
// PrivilegedPublisher — 特权信息 DDS 发布器
// =============================================================================

class PrivilegedPublisher
{
public:
    PrivilegedPublisher() = default;

    // ---- 初始化: 缓存 MuJoCo ID + 创建 DDS Publisher ----
    bool init(mjModel* m)
    {
        // ---- 缓存 MuJoCo body/site/geom IDs ----
        body_left_palm_   = mj_name2id(m, mjOBJ_BODY, "left_palm_link");
        body_right_palm_  = mj_name2id(m, mjOBJ_BODY, "right_palm_link");
        body_left_ankle_  = mj_name2id(m, mjOBJ_BODY, "left_ankle_pitch_link");
        body_right_ankle_ = mj_name2id(m, mjOBJ_BODY, "right_ankle_pitch_link");
        body_head_        = mj_name2id(m, mjOBJ_BODY, "d455_link");
        body_root_        = mj_name2id(m, mjOBJ_BODY, "torso_link");
        body_box_         = mj_name2id(m, mjOBJ_BODY, "box");
        site_goal_        = mj_name2id(m, mjOBJ_SITE, "goal");

        // 箱子 geom (用于读取尺寸)
        if (body_box_ >= 0)
        {
            int geom_start = m->body_geomadr[body_box_];
            int geom_num   = m->body_geomnum[body_box_];
            if (geom_num > 0 && m->geom_type[geom_start] == mjGEOM_BOX)
            {
                geom_box_ = geom_start;
            }
        }

        // 验证关键 body
        const char* required[] = {"torso_link", "left_palm_link", "right_palm_link",
                                  "left_ankle_pitch_link", "right_ankle_pitch_link", "d455_link"};
        int* ids[] = {&body_root_, &body_left_palm_, &body_right_palm_,
                      &body_left_ankle_, &body_right_ankle_, &body_head_};
        for (int i = 0; i < 6; ++i)
        {
            if (*ids[i] < 0)
            {
                printf("[PrivilegedPublisher] WARNING: body '%s' not found in MJCF\n", required[i]);
            }
        }

        // ---- 创建 DDS Publisher ----
        publisher_ = std::make_unique<Publisher_t>(TOPIC_NAME);
        printf("[PrivilegedPublisher] DDS topic '%s' created\n", TOPIC_NAME);
        return true;
    }

    // ---- 发布一帧特权信息 (每个仿真步调用, ~1000Hz) ----
    void publish(mjModel* m, mjData* d)
    {
        if (!publisher_) return;
        if (!publisher_->trylock()) return;

        auto& msg = publisher_->msg_;
        std::memset(&msg, 0, sizeof(msg));
        msg.sequence = sequence_++;

        // (A) 末端执行器世界坐标 (5 bodies × 3)
        copyBodyPos(d, body_left_palm_,   msg.end_effector_pos_w + 0);
        copyBodyPos(d, body_right_palm_,  msg.end_effector_pos_w + 3);
        copyBodyPos(d, body_left_ankle_,  msg.end_effector_pos_w + 6);
        copyBodyPos(d, body_right_ankle_, msg.end_effector_pos_w + 9);
        copyBodyPos(d, body_head_,        msg.end_effector_pos_w + 12);

        // (B) 根Link世界坐标
        copyBodyPos(d, body_root_, msg.root_pos_w);

        // (C) 箱子世界坐标和姿态
        if (body_box_ >= 0)
        {
            msg.box_pos_w[0] = static_cast<float>(d->xpos[body_box_ * 3 + 0]);
            msg.box_pos_w[1] = static_cast<float>(d->xpos[body_box_ * 3 + 1]);
            msg.box_pos_w[2] = static_cast<float>(d->xpos[body_box_ * 3 + 2]);

            // MuJoCo xquat: [w, x, y, z]
            msg.box_quat_w[0] = static_cast<float>(d->xquat[body_box_ * 4 + 0]);
            msg.box_quat_w[1] = static_cast<float>(d->xquat[body_box_ * 4 + 1]);
            msg.box_quat_w[2] = static_cast<float>(d->xquat[body_box_ * 4 + 2]);
            msg.box_quat_w[3] = static_cast<float>(d->xquat[body_box_ * 4 + 3]);
        }

        // (D) 箱子尺寸 (静态, 从 mjModel 读取)
        if (geom_box_ >= 0)
        {
            msg.box_size[0] = static_cast<float>(m->geom_size[geom_box_ * 3 + 0]);
            msg.box_size[1] = static_cast<float>(m->geom_size[geom_box_ * 3 + 1]);
            msg.box_size[2] = static_cast<float>(m->geom_size[geom_box_ * 3 + 2]);
        }

        // (E) 目标点世界坐标
        if (site_goal_ >= 0)
        {
            msg.goal_pos_w[0] = static_cast<float>(d->site_xpos[site_goal_ * 3 + 0]);
            msg.goal_pos_w[1] = static_cast<float>(d->site_xpos[site_goal_ * 3 + 1]);
            msg.goal_pos_w[2] = static_cast<float>(d->site_xpos[site_goal_ * 3 + 2]);
        }

        publisher_->unlockAndPublish();
    }

private:
    using Publisher_t = unitree::robot::RealTimePublisher<PrivilegedStateMsg>;

    std::unique_ptr<Publisher_t> publisher_;
    uint32_t sequence_ = 0;

    // 缓存的 MuJoCo body/site/geom ID
    int body_left_palm_   = -1;
    int body_right_palm_  = -1;
    int body_left_ankle_  = -1;
    int body_right_ankle_ = -1;
    int body_head_        = -1;
    int body_root_        = -1;
    int body_box_         = -1;
    int site_goal_        = -1;
    int geom_box_         = -1;

    static constexpr const char* TOPIC_NAME = "rt/privileged_state";

    // 辅助: 从 d->xpos 拷贝 body 世界坐标
    static void copyBodyPos(mjData* d, int body_id, float* dst)
    {
        if (body_id >= 0)
        {
            dst[0] = static_cast<float>(d->xpos[body_id * 3 + 0]);
            dst[1] = static_cast<float>(d->xpos[body_id * 3 + 1]);
            dst[2] = static_cast<float>(d->xpos[body_id * 3 + 2]);
        }
    }
};
