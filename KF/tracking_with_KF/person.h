#ifndef PERSON_H
#define PERSON_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include "blob.h"

using namespace cv;
using namespace std;

// person class can constrain the relative positions of hands and face
class Person{
private:
    Blob _face;
    Blob _right_hand;
    Blob _left_hand;
    int _occluded;
public:
    Person(){

        _face = Blob(Point(160,60),
                     Size(30,30),
                     0.0,
                     Scalar(0,255,0));
        _right_hand = Blob(Point(90,160),
                          Size(30,30),
                          0.0,
                          Scalar(0,255,255));
        _left_hand = Blob(Point(250,160),
                           Size(30,30),
                           0.0,
                           Scalar(0,0,255));
    }
    Person(Blob face, Blob right_hand, Blob left_hand):
        _face(face),
        _right_hand(right_hand),
        _left_hand(left_hand){
    }
    void clear(){
        _face.clear();
        _right_hand.clear();
        _left_hand.clear();
    }

    void set_face(Blob face){
        _face = face;
    }
    void set_right_hand(Blob right_hand){
        _right_hand = right_hand;
    }

    void set_left_hand(Blob left_hand){
        _left_hand = left_hand;
    }

    Blob get_face(){
        return _face;
    }

    Blob get_right_hand(){
        return _right_hand;
    }

    Blob get_left_hand(){
        return _left_hand;
    }

    void draw(Mat& image, bool draw_ellipses){
        _face.draw(image,draw_ellipses);
        _right_hand.draw(image,draw_ellipses);
        _left_hand.draw(image,draw_ellipses);
    }
    void move(const Point& delta_FACE,
              const Point& delta_RIGHT_HAND,
              const Point& delta_LEFT_HAND){
        _face.move(delta_FACE);
        _right_hand.move(delta_RIGHT_HAND);
        _left_hand.move(delta_LEFT_HAND);
    }

    void predict(){
        _face.predict();
        _right_hand.predict();
        _left_hand.predict();
    }

    void correct(const Mat& faceMeasure,
                 const Mat& rhandMeasure,
                 const Mat& lhandMeasure){
        _face.correct(faceMeasure);
        _right_hand.correct(rhandMeasure);
        _left_hand.correct(lhandMeasure);
    }

};

#endif // PERSON_H
