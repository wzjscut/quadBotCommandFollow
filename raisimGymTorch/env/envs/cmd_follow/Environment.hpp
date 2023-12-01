//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {
  
    curriculum_ = itr_number_;
  
    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    anymal_ = world_->addArticulatedSystem(resourceDir_+"/raibo2/raibo2.urdf");
    anymal_->setName("raibo2");
    anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 36;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
    actionStd_.setConstant(action_std);
    READ_YAML(bool, cmd_follow_flag_, cfg_["cmd_follow_flag"])

    /// reward 
    READ_YAML(double, done_reward_, cfg_["done_reward"])
    READ_YAML(double, vx_reward_coeff_, cfg_["vx_reward_coeff"])
    READ_YAML(double, wz_reward_coeff_, cfg_["wz_reward_coeff"])
    READ_YAML(double, torque_coeff_, cfg_["torque_coeff"])
    
    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(anymal_);
    }

  }

  void init() final { }

  void sample_target() {
    double vx_vec[] = {0.0, 1.0, 1.0, 1.0};
    double wz_vec[] = {0.0, -0.6, 0.6, -0.6, 0.6};

    if (true == cmd_follow_flag_)
    {
      target_vx_ = cmd_vx_;
      target_wz_ = cmd_wz_;
    }
    else
    {
      target_vx_ = vx_vec[std::rand() % 4];
      target_wz_ = wz_vec[std::rand() % 3];
    }

    if (visualizable_) {
      std::cout << "sample vx = " << target_vx_ << ", wz = " << target_wz_ << std::endl;
    }
  }

  void reset() final {
    step_ = 0;
    anymal_->setState(gc_init_, gv_init_);
    sample_target();

    for (int i = 0; i < 50; i++)
    {
      if (server_)
        server_->lockVisualizationServerMutex();
      world_->integrate();
      if (server_)
        server_->unlockVisualizationServerMutex();
    }

    updateObservation();
    updateObservation();
    updateObservation();
  }

  double compute_forward_reward()
  {
    double r = 0.;
    double dvx = abs(target_vx_ - bodyLinearVel_[0]);
    double dwz = abs(target_wz_ - bodyAngularVel_[2]);
    double dvyz = abs(bodyLinearVel_[1]) + abs(bodyLinearVel_[2]);
    double dwxy = abs(bodyAngularVel_[0]) + abs(bodyAngularVel_[1]);

    r += vx_reward_coeff_ * (4.0 - 2.0 * (dvx + dwz));
    if (abs(target_vx_) < 0.01 && abs(target_wz_) < 0.01)  // stop mode
    {
      r += -vx_reward_coeff_ * 0.5 * dvyz;
      r += -wz_reward_coeff_ * 0.5 * dwxy;
    }
    else if (abs(target_vx_) < 0.01)  // stop mode
    {
      r += -vx_reward_coeff_ * 0.5 * dvyz;
    }
    else if (abs(target_wz_) < 0.01)  // stop mode
    {
      r += -wz_reward_coeff_ * 0.5 * dwxy;
    }
    else // mixture
    {
      r += -vx_reward_coeff_ * 0.5 * dvyz;
      r += -wz_reward_coeff_ * 0.5 * dwxy;
    }

    return r;
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    step_++;
    if (curriculum_ > 20000 && step_ % 100 == 5)
    {
      sample_target();
    }

    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    anymal_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    updateObservation();

    torque_reward_ = torque_coeff_ * anymal_->getGeneralizedForce().squaredNorm();
    speed_reward_ = compute_forward_reward();
    all_reward_ = torque_reward_ + speed_reward_;
    return all_reward_;
  }

  void updateObservation() {
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    obDouble_ << gc_[2], /// body height 1 dof
        rot.e().row(2).transpose(), /// body orientation 3 dof
        gc_.tail(12), /// joint angles 12 dof
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity 6 dof
        gv_.tail(12), /// joint velocity 12 dof
        target_vx_, target_wz_; /// target 2dof
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(done_reward_);

    /// if the contact body is not feet
    for(auto& contact: anymal_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        return true;

    terminalReward = 0.f;
    return false;
  }

  void getRewardInfo(Eigen::Ref<EigenVec> reward)
  {
    reward << 
    abs(target_vx_ - bodyLinearVel_[0]), 
    abs(target_wz_ - bodyAngularVel_[2]), 
    abs(bodyLinearVel_[1]) + abs(bodyLinearVel_[2]), 
    abs(bodyAngularVel_[0]) + abs(bodyAngularVel_[1]), 
    anymal_->getGeneralizedForce().squaredNorm(), 
    speed_reward_, 
    all_reward_ ,
    0,0,0,0,0,0,0,0,0;
    return;
  }

  void curriculumUpdate() {
    curriculum_++;
  };

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* anymal_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  double done_reward_=0.0, vx_reward_coeff_=0.0, wz_reward_coeff_=0.0, torque_coeff_=0.0;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;

  bool cmd_follow_flag_ = false;
  double target_vx_ = 0.0;
  double target_wz_ = 0.0;
  int step_ = 0, curriculum_ = 0;
  double torque_reward_ = 0.0;
  double speed_reward_ = 0.0;
  double all_reward_ = 0.0;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

