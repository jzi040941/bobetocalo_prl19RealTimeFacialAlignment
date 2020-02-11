#include <FaceHeadPoseOpencv.hpp>
#include <trace.hpp>
#include <utils.hpp>
#include <ModernPosit.h>
#include <boost/program_options.hpp>
#include "tensorflow/cc/ops/standard_ops.h"
using namespace cv::dnn;
namespace upm {


  void FaceHeadPoseOpencv::parseOptions
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
  UPM_PRINT(desc);
};
 
void
  FaceHeadPoseOpencv::train
    (
    const std::vector<FaceAnnotation> &anns_train,
    const std::vector<FaceAnnotation> &anns_valid
    ){

    }


  void
FaceHeadPoseOpencv::load()
{
  /* 
    if (not face_cascade.load(_cascade_name)) {
      std::cerr << "Cannot load cascade classifier: " << _cascade_name << std::endl;
    }
  */
 }

void
FaceHeadPoseOpencv::process
  (
  cv::Mat frame,
  std::vector<FaceAnnotation> &faces,
  const FaceAnnotation &ann
  )
{
  /// init factors
  
  //std::vector<Point2f> projectionOutput(objectPointsForReprojection.size());
  Size small_size(700, 700 * (float) frame.rows / (float) frame.cols);
  scaleFactor = 700.0f / frame.cols;

  w = small_size.width;
  h = small_size.height;
  Matx33f K {w, 0, w/2.0f,
             0, w, h/2.0f,
             0, 0, 1.0f};

  
  ///ModernPosit init
  std::string positpath = _path;
  std::vector<unsigned int> mask{1,2,3,4,5,6,7,101,8,11,102,12,15,16,17,18,19,20,103,21,24};
  
  double data[9] = {(double)frame.cols, 0, (double)frame.cols/2.0f,
             0, (double)frame.cols, (double)frame.rows/2.0f,
             0, 0, 1.0f};
  
  
  /*
  double data[9] = {1.0f, 0, 1.0/2.0f,
             0, 1.0f, 1.0/2.0f,
             0, 0, 1.0f};
*/
  cv::Mat cam_matrix = cv::Mat(3, 3, CV_64F, data);
  
  std::cout <<"cam_matrix"<<cam_matrix<<std::endl;
  std::vector<cv::Point3f> world_all;
  std::vector<unsigned int> index_all;
  std::vector<cv::Point3f> world_pts;
  std::vector<cv::Point2f> image_pts;
  ModernPosit::loadWorldShape(positpath,mask,world_all, index_all);


  /// Anlyze each detected face
  for (FaceAnnotation &face : faces)
  {
    ///

    std::vector<Point2f> points2d;
    points2d.push_back(face.parts[FacePartLabel::leye].landmarks[0].pos/scaleFactor);
    points2d.push_back(face.parts[FacePartLabel::reye].landmarks.back().pos/scaleFactor);
    points2d.push_back(face.parts[FacePartLabel::nose].landmarks[1].pos/scaleFactor);
    points2d.push_back(face.parts[FacePartLabel::tmouth].landmarks.back().pos/scaleFactor);
    points2d.push_back(face.parts[FacePartLabel::tmouth].landmarks[0].pos/scaleFactor);
   
    
     ModernPosit::setCorrespondences(world_all, index_all, face, mask, world_pts, image_pts);
    
     std::cout<<"image_pts:[";
     for(auto pts : image_pts){
        std::cout<<pts<<",";
     }
     std::cout<<"]"<<std::endl;

     std::cout<<"world_pts:[";
     for(auto pts : world_pts){
        std::cout<<pts<<",";
     }
     std::cout<<"]"<<std::endl;
    
    
    cv::Mat rot_matrix(3,3,CV_64F);
    cv::Mat trl_matrix(3,3,CV_64F);
    ModernPosit::run(world_pts,image_pts,cam_matrix ,2,rot_matrix,trl_matrix);
    face.headpose = ModernPosit::rotationMatrixToEuler(rot_matrix);
    
    /*
    vector<Point2f> points2d;
    for (int pId : landmarksIDsFor3DPoints) {
        points2d.push_back(shapes[0][pId] / scaleFactor);
    }
    */

    // Find object/camera transform
    cv::solvePnP(objectPoints, points2d, K, Mat(), rvec, tvec, true);
    
    // Reproject the axes back onto the image
    projectPoints(objectPointsForReprojection, rvec, tvec, K, Mat(), projectionOutput);
    Mat(projectionOutput) *= scaleFactor;
    
    cv::Mat rotM(3,3,CV_64F);
    cv::Rodrigues(rvec,rotM);
    
    cv::Mat ProjectionM(3,4,CV_64F);
    cv::hconcat(rotM,tvec,ProjectionM);
    
    cv::Mat Kd(3, 3, CV_64F); // intrinsic parameter matrix
    cv::Mat Rd(3, 3, CV_64F); // rotation matrix
    cv::Mat Td(4, 1, CV_64F); // translation vector
    cv::Mat rotMx(3, 3, CV_64F); // rotation matrix
    cv::Mat rotMy(3, 3, CV_64F); // rotation matrix
    cv::Mat rotMz(3, 3, CV_64F); // rotation matrix

    cv::Mat Angles(3, 1, CV_64F);
    cv::decomposeProjectionMatrix(ProjectionM, Kd, Rd, Td, rotMx, rotMy, rotMz,Angles);
    //face.headpose = cv::Point3f(Angles.at<double>(0,0), Angles.at<double>(1,0), Angles.at<double>(2,0));
    //face.headpose = cv::Point3f(, Angles.at<float>(1,0), Angles.at<float>(2,0));
    
    arrowedLine(frame, projectionOutput[0], projectionOutput[1], Scalar(255,255,0), 2, 8, 0, 0.3);
                arrowedLine(frame, projectionOutput[0], projectionOutput[2], Scalar(0,255,255), 2, 8, 0, 0.3);
                arrowedLine(frame, projectionOutput[0], projectionOutput[3], Scalar(255,0,255), 2, 8, 0, 0.3);

    std::cout<<face.headpose<<std::endl; 
    break;
  }

}

}
