#include "myslam/frame.h"
#include "myslam/common_include.h"


namespace myslam
{
Frame::Frame()
: id_(-1), time_stamp_(-1), camera_(nullptr), is_key_frame_(false)
{

}

Frame::Frame ( long id, double time_stamp, SE3 T_c_w, Camera::Ptr camera, Mat color, Mat depth )
: id_(id), time_stamp_(time_stamp), T_c_w_(T_c_w), camera_(camera), color_(color) , is_key_frame_(false)
{

}

Frame::~Frame(){}

Frame::Ptr Frame::createFrame()
{
    static long factory_id = 0;			//注意这个static变量,这是个只加不减，记录总数的id
    return Frame::Ptr( new Frame(factory_id++) );
}


void Frame::setPose ( const SE3& T_c_w )
{
    T_c_w_ = T_c_w;
}


Vector3d Frame::getCamCenter() const
{
    return T_c_w_.inverse().translation();
}

bool Frame::isInFrame ( const Vector3d& pt_world )			//判断该点可以被当前帧看到
{
    Vector3d p_cam = camera_->world2camera( pt_world, T_c_w_ );
    // cout<<"P_cam = "<<p_cam.transpose()<<endl;
    if ( p_cam(2,0)<0 ) return false;
    Vector2d pixel = camera_->world2pixel( pt_world, T_c_w_ );		//投影到像素坐标系，看是否落在图像的框内
    // cout<<"P_pixel = "<<pixel.transpose()<<endl<<endl;
    return pixel(0,0)>0 && pixel(1,0)>0 
        && pixel(0,0)<color_.cols 
        && pixel(1,0)<color_.rows;
}

void Frame::UpdateThePosition()
{
      Mat Vec_R;
      Vec_R =  (Mat_<double>(3, 1) << RT_Mat_to_ceres.at<double>(0,0), RT_Mat_to_ceres.at<double>(1,0), RT_Mat_to_ceres.at<double>(2,0)	);
      Mat Mat_R;
      Rodrigues(Vec_R, Mat_R);
      
      
      RT_Mat = (Mat_<double>(4, 4) << 	Mat_R.at<double>(0,0), Mat_R.at<double>(0,1), Mat_R.at<double>(0,2),RT_Mat_to_ceres.at<double>(0,3),//这个是相对于上一帧的RT阵
								Mat_R.at<double>(1,0), Mat_R.at<double>(1,1), Mat_R.at<double>(1,2),RT_Mat_to_ceres.at<double>(1,3),
								Mat_R.at<double>(2,0), Mat_R.at<double>(2,1), Mat_R.at<double>(2,2),RT_Mat_to_ceres.at<double>(2,3),
								0,0,0,1);    
      
      Matrix3d eigen_R ;
      eigen_R<< Mat_R.at<double>(0,0), Mat_R.at<double>(0,1), Mat_R.at<double>(0,2),
			Mat_R.at<double>(1,0), Mat_R.at<double>(1,1), Mat_R.at<double>(1,2),
			Mat_R.at<double>(2,0), Mat_R.at<double>(2,1), Mat_R.at<double>(2,2);
	
      Vector3d eigen_t;
      eigen_t<<RT_Mat_to_ceres.at<double>(0,3), RT_Mat_to_ceres.at<double>(1,3), RT_Mat_to_ceres.at<double>(2,3);
      T_c_w_ = SE3(eigen_R, eigen_t);
}

    void Frame::Initial_the_value()
    {
        Mat R  = Mat::eye(3,3,CV_64FC1);
        Mat t = (Mat_<double>(3,1) << 0, 0,  0);		//这个东西先这样，回头再扔进camera类
        RT_Mat = (Mat_<double>(4, 4) << 	R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),t.at<double>(0,0),//这个是相对于上一帧的RT阵
                R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),t.at<double>(1,0),
                R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2),t.at<double>(2,0),
                0,0,0,1);
        Mat R_vector ;
        Rodrigues(R, R_vector);
        RT_Mat_to_ceres = Mat(6, 1, CV_64FC1);
        R_vector.copyTo(RT_Mat_to_ceres.rowRange(0, 3));
        t.copyTo(RT_Mat_to_ceres.rowRange(3, 6));

        Matrix3d eigen_R = Matrix3d::Identity();
        Vector3d eigen_t ;
        eigen_t << 0, 0, 0;
        T_c_w_ = SE3(eigen_R, eigen_t);			//ATTENTION 注意cw是谁相对于谁！！！Debug的时候一定要注意

    }

}
