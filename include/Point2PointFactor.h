/**
 * @file Point2PointFactor.h
 * @brief 3D Point to Point factor 
 */

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <cmath>


// you can custom namespace (if needed for your project)
namespace gtsam {


class GTSAM_EXPORT Point2PointFactor: public gtsam::NoiseModelFactorN<gtsam::Pose3> {

private:
  // measurement information
  double mx_, my_;
  Eigen::Vector3d msrc_, mtgt_;

  typedef NoiseModelFactorN<Pose3> Base;

public:
  using Base::evaluateError;

  typedef std::shared_ptr<Point2PointFactor> shared_ptr;

  /// Typedef to this class
  typedef Point2PointFactor This;

  Point2PointFactor() {}

  ~Point2PointFactor() override {}

  /**
   * Constructor
   * @param poseKey  associated pose varible key, identical id
   * @param src      Point3 measurement
   * @param tgt      Point3 measurement
   * @param model    noise model
   */
  Point2PointFactor(gtsam::Key poseKey, const Eigen::Vector3d src, Eigen::Vector3d tgt, const gtsam::SharedNoiseModel& model) :
      Base(model, poseKey), msrc_(src), mtgt_(tgt) {}


  /// @return a deep copy of this factor
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }


  // error function, most imprtant, decide the error item in the cost function
  // @param p    the pose in Pose2
  // @param H    the optional Jacobian matrix, which use boost optional and has default null pointer
  Vector evaluateError(const Pose3 &X, OptionalMatrixType H) const {
    const gtsam::Rot3 &R = X.rotation();
    gtsam::Vector3 mx = R * msrc_ + X.translation();
    gtsam::Vector3 error = mx - mtgt_;
    if (H) {
      *H = gtsam::Matrix(3, 6);
      (*H).block(0, 0, 3, 3) = -X.rotation().matrix() * skewSymmetric(msrc_);
      (*H).block(0, 3, 3, 3) = X.rotation().matrix();
    }
    return error;
  }

};
} // namespace gtsam
