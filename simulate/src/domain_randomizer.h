#pragma once

// =============================================================================
// DomainRandomizer — 仿真环境域随机化
//
// 在每次 reset 时随机化:
//   - 箱子位置 (x,y,z) + 偏航角 (yaw)
//   - 箱子尺寸 (宽/深/高)
//   - 目标点位置 (x,y,z)
//
// 使用方式:
//   domain_randomizer.init(m);        // 模型加载后调用一次
//   domain_randomizer.randomize(m,d); // 每次 reset 后调用 (mj_resetData 之后, mj_forward 之前)
// =============================================================================

#include <mujoco/mujoco.h>

#include <cmath>
#include <cstdio>
#include <random>


class DomainRandomizer
{
public:
    DomainRandomizer() : rng_(std::random_device{}()) {}

    // ---- 初始化: 缓存 MuJoCo ID (模型加载后调用一次) ----
    bool init(mjModel* m)
    {
        box_body_id_  = mj_name2id(m, mjOBJ_BODY, "box");
        goal_site_id_ = mj_name2id(m, mjOBJ_SITE, "goal");

        if (box_body_id_ < 0)
        {
            printf("[DomainRandomizer] WARNING: 'box' body not found in MJCF, "
                   "randomization disabled\n");
            initialized_ = false;
            return false;
        }

        // 找到 box body 的 free joint
        int joint_start = m->body_jntadr[box_body_id_];
        int joint_num   = m->body_jntnum[box_body_id_];
        int joint_id    = -1;
        for (int i = 0; i < joint_num; i++)
        {
            if (m->jnt_type[joint_start + i] == mjJNT_FREE)
            {
                joint_id = joint_start + i;
                break;
            }
        }
        if (joint_id < 0)
        {
            printf("[DomainRandomizer] WARNING: 'box' body has no free joint\n");
            initialized_ = false;
            return false;
        }
        box_qpos_adr_ = m->jnt_qposadr[joint_id];

        // 缓存 box geom (用于尺寸随机化)
        int geom_start = m->body_geomadr[box_body_id_];
        int geom_num   = m->body_geomnum[box_body_id_];
        if (geom_num > 0 && m->geom_type[geom_start] == mjGEOM_BOX)
        {
            box_geom_id_ = geom_start;
        }

        if (goal_site_id_ < 0)
        {
            printf("[DomainRandomizer] WARNING: 'goal' site not found in MJCF\n");
        }

        initialized_ = true;
        printf("[DomainRandomizer] Initialized (box_body=%d, box_qpos_adr=%d, "
               "box_geom=%d, goal_site=%d)\n",
               box_body_id_, box_qpos_adr_, box_geom_id_, goal_site_id_);
        return true;
    }

    // ---- 随机化: 在 mj_resetData 之后、mj_forward 之前调用 ----
    void randomize(mjModel* m, mjData* d)
    {
        if (!initialized_) return;

        // ---- (A) 箱子位置 ----
        double box_x = uniform(BOX_X_MIN, BOX_X_MAX);
        double box_y = uniform(BOX_Y_MIN, BOX_Y_MAX);
        double box_z = uniform(BOX_Z_MIN, BOX_Z_MAX);

        // ---- (B) 箱子偏航角 (仅绕Z轴旋转) ----
        double yaw = uniform(BOX_YAW_MIN, BOX_YAW_MAX);
        double qw  = std::cos(yaw / 2.0);
        double qz  = std::sin(yaw / 2.0);

        // free joint qpos: [x, y, z, qw, qx, qy, qz]
        int adr = box_qpos_adr_;
        d->qpos[adr + 0] = box_x;
        d->qpos[adr + 1] = box_y;
        d->qpos[adr + 2] = box_z;
        d->qpos[adr + 3] = qw;
        d->qpos[adr + 4] = 0.0;
        d->qpos[adr + 5] = 0.0;
        d->qpos[adr + 6] = qz;

        // ---- (C) 箱子尺寸 (MuJoCo geom_size 是半边长) ----
        if (box_geom_id_ >= 0)
        {
            double half_w = uniform(BOX_WIDTH_MIN  / 2.0, BOX_WIDTH_MAX  / 2.0);
            double half_d = uniform(BOX_DEPTH_MIN  / 2.0, BOX_DEPTH_MAX  / 2.0);
            double half_h = uniform(BOX_HEIGHT_MIN / 2.0, BOX_HEIGHT_MAX / 2.0);
            m->geom_size[box_geom_id_ * 3 + 0] = half_w;
            m->geom_size[box_geom_id_ * 3 + 1] = half_d;
            m->geom_size[box_geom_id_ * 3 + 2] = half_h;
        }

        // ---- (D) 目标点位置 ----
        if (goal_site_id_ >= 0)
        {
            m->site_pos[goal_site_id_ * 3 + 0] = uniform(GOAL_X_MIN, GOAL_X_MAX);
            m->site_pos[goal_site_id_ * 3 + 1] = uniform(GOAL_Y_MIN, GOAL_Y_MAX);
            m->site_pos[goal_site_id_ * 3 + 2] = uniform(GOAL_Z_MIN, GOAL_Z_MAX);
        }

        printf("[DomainRandomizer] box=(%.2f,%.2f,%.2f) yaw=%.0f° "
               "size=(%.2f,%.2f,%.2f) goal=(%.2f,%.2f,%.2f)\n",
               box_x, box_y, box_z, yaw * 180.0 / M_PI,
               m->geom_size[box_geom_id_ * 3 + 0] * 2,
               m->geom_size[box_geom_id_ * 3 + 1] * 2,
               m->geom_size[box_geom_id_ * 3 + 2] * 2,
               m->site_pos[goal_site_id_ * 3 + 0],
               m->site_pos[goal_site_id_ * 3 + 1],
               m->site_pos[goal_site_id_ * 3 + 2]);
    }

private:
    std::mt19937 rng_;
    bool initialized_ = false;

    // 缓存的 MuJoCo ID
    int box_body_id_   = -1;
    int box_qpos_adr_  = -1;
    int box_geom_id_   = -1;
    int goal_site_id_  = -1;

    // =========================================================================
    // 随机化范围 (可根据训练需要调整)
    // =========================================================================

    // 箱子位置 (世界坐标系)
    static constexpr double BOX_X_MIN   = 0.3;
    static constexpr double BOX_X_MAX   = 0.8;
    static constexpr double BOX_Y_MIN   = -0.3;
    static constexpr double BOX_Y_MAX   = 0.3;
    static constexpr double BOX_Z_MIN   = 0.2;
    static constexpr double BOX_Z_MAX   = 0.5;

    // 箱子偏航角 (仅绕Z轴)
    static constexpr double BOX_YAW_MIN = 0.0;
    static constexpr double BOX_YAW_MAX = 2.0 * M_PI;

    // 箱子尺寸 (全边长, 非半边长)
    static constexpr double BOX_WIDTH_MIN  = 0.15;
    static constexpr double BOX_WIDTH_MAX  = 0.25;
    static constexpr double BOX_DEPTH_MIN  = 0.15;
    static constexpr double BOX_DEPTH_MAX  = 0.25;
    static constexpr double BOX_HEIGHT_MIN = 0.1;
    static constexpr double BOX_HEIGHT_MAX = 0.2;

    // 目标点位置 (世界坐标系)
    static constexpr double GOAL_X_MIN  = 0.5;
    static constexpr double GOAL_X_MAX  = 1.5;
    static constexpr double GOAL_Y_MIN  = -0.5;
    static constexpr double GOAL_Y_MAX  = 0.5;
    static constexpr double GOAL_Z_MIN  = 0.0;
    static constexpr double GOAL_Z_MAX  = 0.3;

    double uniform(double min_val, double max_val)
    {
        return std::uniform_real_distribution<double>(min_val, max_val)(rng_);
    }
};
