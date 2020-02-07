#include <FaceDetectorOpencv.hpp>
#include <trace.hpp>
#include <utils.hpp>
#include <ModernPosit.h>
#include <boost/program_options.hpp>
#include "tensorflow/cc/ops/standard_ops.h"
using namespace cv::dnn;
namespace upm {

  const float BBOX_SCALE = 0.3f;
  const cv::Size FACE_SIZE = cv::Size(256,256);
  const cv::Scalar meanVal(104.0, 177.0, 123.0);
  
  
void FaceDetectorOpencv::detectFaceOpenCVDNN(const cv::Mat& frameOpenCVDNN,  std::vector<FaceAnnotation> &faces,
    Net net)
{    
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
#ifdef CAFFE
        cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
#else
        cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);
#endif

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    faces.clear();
    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > confidenceThreshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
            
            faces.push_back(upm::FaceAnnotation());
            faces[i].bbox.pos = cv::Rect2f(x1,y1,x2-x1,y2-y1);
            std::cout<<"xyxy"<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<std::endl;
            //faces.push_back(Rect(Point2i(x1,y1),Point2i(x2,y2)));
            //cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
        }
    }



}

void FaceDetectorOpencv::faceDetector(const cv::Mat& image,
                  std::vector<FaceAnnotation> &faces,
                  cv::CascadeClassifier &face_cascade) {
  cv::Mat gray;

    // The cascade classifier works best on grayscale images
    if (image.channels() > 1) {
        cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Histogram equalization generally aids in face detection
    equalizeHist(gray, gray);

    //faces.clear();
    
    std::vector<Rect> tempfaces;
    // Run the cascade classifier
    face_cascade.detectMultiScale(gray, tempfaces, 1.4, 3, CASCADE_SCALE_IMAGE + CASCADE_FIND_BIGGEST_OBJECT);
    
    if(!tempfaces.empty()){
      faces.push_back(upm::FaceAnnotation());
      faces[0].bbox.pos = tempfaces[0];
          }
       
}
    void FaceDetectorOpencv::parseOptions
  (
  int argc,
  char **argv
  )
{
  /// Declare the supported program options
  //upm::FaceDetector::parseOptions(argc, argv);
  namespace po = boost::program_options;
  po::options_description desc("FaceDetectorOpencv options");
  po::variables_map vm;
  desc.add_options()
    ("face_ssd_config_file", po::value<std::string>()->default_value("opencv_face_detector.pbtxt"), "Path to face cascade(ssd model pbtxt)")
    ("face_ssd_weight_file", po::value<std::string>()->default_value("opencv_face_detector_uint8.pb"), "Path to face cascade(ssd model pbfile)")
     ("cascade_name", po::value<std::string>()->default_value("haarcascade_frontalface_default.xml"), "Path to face cascade(xml file)");


  po::command_line_parser parser(argc, argv);
  parser.options(desc);
  const po::parsed_options parsed_opt(parser.allow_unregistered().run());
  po::store(parsed_opt, vm);
  po::notify(vm);


  if (vm.count("face_ssd_config_file"))
    _faceDetectorConfigFile = _path+vm["face_ssd_config_file"].as<std::string>();
  if (vm.count("face_ssd_weight_file"))
    _faceDetectorWeightFile = _path+vm["face_ssd_weight_file"].as<std::string>();
  if (vm.count("cascade_name"))
    _cascade_name = _path+vm["cascade_name"].as<std::string>();


  UPM_PRINT(desc);
};
 
void
  FaceDetectorOpencv::train
    (
    const std::vector<FaceAnnotation> &anns_train,
    const std::vector<FaceAnnotation> &anns_valid
    ){

    }


  void
FaceDetectorOpencv::load()
{
  std::cout<<"configfile:"<<_faceDetectorConfigFile<<std::endl;
  /* 
    if (not face_cascade.load(_cascade_name)) {
      std::cerr << "Cannot load cascade classifier: " << _cascade_name << std::endl;
    }
  */
  
  net = cv::dnn::readNetFromTensorflow(_faceDetectorWeightFile, _faceDetectorConfigFile);
  net.setPreferableBackend(DNN_BACKEND_CUDA);
  net.setPreferableTarget(DNN_TARGET_CUDA);
  
}

void
FaceDetectorOpencv::process
  (
  cv::Mat frameOpenCVDNN,
  std::vector<FaceAnnotation> &faces,
  const FaceAnnotation &ann
  )
{
  detectFaceOpenCVDNN(frameOpenCVDNN,  faces, net);
  //faceDetector(frameOpenCVDNN,  faces, face_cascade);
  
}

}
