#ifndef BLOB_H
#define BLOB_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

using namespace std;
using namespace cv;

class Blob{
private:
    Point _center;
    Size _axes;
    double _angle;
    Scalar _color;
    bool _occluded;
    vector<Point> _blob_points;
    KalmanFilter _kalman;
    Mat _predicted_state;
    Mat _estimated_state;
protected:
    void initKalmanFilter(){
        _kalman.init(4,2,0);
        _kalman.transitionMatrix = *(Mat_<float>(4, 4)
                                     << 1,0,1,0,
                                     0,1,0,1,
                                     0,0,1,0,
                                     0,0,0,1);

        setIdentity(_kalman.measurementMatrix);
        setIdentity(_kalman.processNoiseCov, Scalar::all(1e2));
        setIdentity(_kalman.measurementNoiseCov, Scalar::all(1e-10));
        setIdentity(_kalman.errorCovPost, Scalar::all(.1));

    }

public:
    Blob(){
        _center = Point(100,100);
        _axes = Size(50,30);
        _angle = 0;
        _color = Scalar(255);
        _occluded = false;
        initKalmanFilter();
        _kalman.statePre.at<float>(0) = 100.0;
        _kalman.statePre.at<float>(1) = 100.0;
        _kalman.statePre.at<float>(2) = 0.0;
        _kalman.statePre.at<float>(3) = 0.0;
    }
    Blob(Point center,
         Size size,
         double angle = 0.0,
         Scalar color = Scalar(0),
         bool occluded = false):
        _center(center),
        _axes(size),
        _angle(angle),
        _color(color),
        _occluded(occluded){
        initKalmanFilter();
        _kalman.statePre.at<float>(0) = (float)center.x;
        _kalman.statePre.at<float>(1) = (float)center.y;
        _kalman.statePre.at<float>(2) = 0.0;
        _kalman.statePre.at<float>(3) = 0.0;

    }

    Blob(const RotatedRect& rect,
         const vector<Point>& points,
         Scalar color,
         bool occluded = false):
        _axes(Size(rect.size.width,rect.size.height)),
        _angle(rect.angle),
        _color(color),
        _blob_points(points),
        _occluded(occluded){
        _center = Point((int)rect.center.x,(int)rect.center.y);
    }

    void clear(){
        _blob_points.clear();
    }

    Point get_blob_center(){
        return _center;
    }

    Size get_blob_axes(){
        return _axes;
    }

    double get_angle(){
        return _angle;
    }

    Mat get_prediction(){
        return _predicted_state;
    }

    Mat get_estimated(){
        return _estimated_state;
    }

    void update(Point center){
        _center = center;
    }

     void predict(){
        _predicted_state = _kalman.predict();
        //return prediction;
    }

     void correct(const Mat& measurement){
        _estimated_state = _kalman.correct(measurement);
        int x = static_cast<int>(_estimated_state.at<float>(0));
        int y = static_cast<int>(_estimated_state.at<float>(1));
        _center = Point(x,y);
     }

    void move(Point delta){
        _center += delta;
    }

    void draw(Mat& image, bool draw_ellipse = false){
        if (!_occluded)
            if (draw_ellipse){
                ellipse(image, _center,_axes,_angle,0,360,_color,2);
            }
            else{
                for(int i = 0; i < _blob_points.size(); i++){
                    Point p(_blob_points[i].y,_blob_points[i].x);
                    if (image.channels() == 1)
                        image.at<uchar>(p.x,p.y) = _color[0];
                    else{
                        image.at<Vec3b>(p.x,p.y)[0] = _color[0];
                        image.at<Vec3b>(p.x,p.y)[1] = _color[1];
                        image.at<Vec3b>(p.x,p.y)[2] = _color[2];
                    }
                }
            }

    }
    void draw(Mat& image, bool draw_ellipse, Scalar color){
        _color = color;
        if (!_occluded)
            if (draw_ellipse){
                ellipse(image, _center,_axes,_angle,0,360,_color,2);
            }
            else{
                for(int i = 0; i < _blob_points.size(); i++){
                    Point p(_blob_points[i].y,_blob_points[i].x);
                    if (image.channels() == 1)
                        image.at<uchar>(p.x,p.y) = _color[0];
                    else{
                        image.at<Vec3b>(p.x,p.y)[0] = _color[0];
                        image.at<Vec3b>(p.x,p.y)[1] = _color[1];
                        image.at<Vec3b>(p.x,p.y)[2] = _color[2];
                    }
                }
            }

    }

};

#endif // BLOB_H
