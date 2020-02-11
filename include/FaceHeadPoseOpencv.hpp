/** ****************************************************************************
 *  @file    FaceHeadPoseOpencv.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2019/04
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_HEAD_POSE_OPENCV_HPP
#define FACE_HEAD_POSE_OPENCV_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FaceHeadPose.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
using namespace cv::dnn;
using namespace cv;
namespace upm {

/** ****************************************************************************
 * @class FaceHeadPoseOpencv
 * @brief Class used for facial feature point detection.
 ******************************************************************************/
class FaceHeadPoseOpencv: public FaceHeadPose
{
public:
  FaceHeadPoseOpencv(std::string path) : _path(path) {};

  ~FaceHeadPoseOpencv() {};
  
  void detectFaceOpenCVDNN(const cv::Mat& frameOpenCVDNN,  std::vector<FaceAnnotation> &faces,
    Net net);

  void
  parseOptions
    (
    int argc,
    char **argv
    );
  
   void
  train
    (
    const std::vector<FaceAnnotation> &anns_train,
    const std::vector<FaceAnnotation> &anns_valid
    );

  void
  load();

  void
  process
    (
    cv::Mat frame,
    std::vector<FaceAnnotation> &faces,
    const FaceAnnotation &ann
    );
  
private:
  std::vector<int> landmarksIDsFor3DPoints {45, 36, 30, 48, 54};
  std::vector<Point3f> objectPoints {
        {8.27412, 1.33849, 10.63490},    //left eye corner
        {-8.27412, 1.33849, 10.63490},   //right eye corner
        {0, -4.47894, 17.73010},         //nose tip
        {-4.61960, -10.14360, 12.27940}, //right mouth corner
        {4.61960, -10.14360, 12.27940},  //left mouth corner
  };
  std::vector<Point3f> objectPointsForReprojection {
          objectPoints[2],                   // nose
          objectPoints[2] + Point3f(0,0,15), // nose and Z-axis
          objectPoints[2] + Point3f(0,15,0), // nose and Y-axis
          objectPoints[2] + Point3f(15,0,0)  // nose and X-axis
  };

  std::vector<Point2f> projectionOutput;

  float scaleFactor;

  float w;
  float h;
  cv::Mat rvec = Mat::zeros(3, 1, CV_64FC1);
  cv::Mat tvec = Mat::zeros(3, 1, CV_64FC1);


  std::string _path;

};

} // namespace upm

#endif /* FACE__PRL19_HPP */
