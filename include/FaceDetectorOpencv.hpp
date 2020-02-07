/** ****************************************************************************
 *  @file    FaceDetectorOpencv.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2019/04
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_DETECTOR_OPENCV_HPP
#define FACE_DETECTOR_OPENCV_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FaceDetector.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
using namespace cv::dnn;
using namespace cv;
namespace upm {

/** ****************************************************************************
 * @class FaceDetectorOpencv
 * @brief Class used for facial feature point detection.
 ******************************************************************************/
class FaceDetectorOpencv: public FaceDetector
{
public:
  FaceDetectorOpencv(std::string path) : _path(path) {};

  ~FaceDetectorOpencv() {};
  
  void detectFaceOpenCVDNN(const cv::Mat& frameOpenCVDNN,  std::vector<FaceAnnotation> &faces,
    Net net);

  void faceDetector(const cv::Mat& image,
                  std::vector<FaceAnnotation> &faces,
                  cv::CascadeClassifier &face_cascade);

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
    cv::Mat frameOpenCVDNN,
    std::vector<FaceAnnotation> &faces,
    const FaceAnnotation &ann
    );
  
  const size_t inWidth = 300;
  const size_t inHeight = 300;
  const double inScaleFactor = 1.0;
  const float confidenceThreshold = 0.7;
  cv::dnn::Net net;

  cv::CascadeClassifier face_cascade;
private:
  std::string _path;
  std::string _faceDetectorConfigFile;
  std::string _faceDetectorWeightFile;
  std::string _cascade_name;
  std::vector<unsigned int> _cnn_landmarks;
  

};

} // namespace upm

#endif /* FACE_ALIGNMENT_PRL19_HPP */
