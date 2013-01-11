#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <vector>
#include <dirent.h>
#include "blob.h"
#include "person.h"
#include "tracker.h"

#define N 10

using namespace std;
using namespace cv;

RNG rng(0xFFFFFFFF);
bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;

Mat image;
Point origin;
Rect selection;
struct Box{
    Rect rect;
    Scalar color;
};

std::vector<Box> boxes;
int vmin = 75, vmax = 256, smin = 100;
Box box;

void onMouse( int event, int x, int y, int, void* )
{
    if( selectObject )
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);

        selection &= Rect(0, 0, image.cols, image.rows);
    }

    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN:
        origin = Point(x,y);
        selection = Rect(x,y,0,0);
        selectObject = true;
        break;
    case CV_EVENT_LBUTTONUP:
        selectObject = false;
        if( selection.width > 0 && selection.height > 0 ){
            trackObject = -1;

            box.rect = selection;
            box.color =  Scalar::all(0); //Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));

            boxes.insert(boxes.end(),box);
        }
        break;
    }
}

Mat binarize_x(const Mat& frame){
    Mat gray;
    vector<Rect> faces;
    CascadeClassifier cascade;
    cascade.load("haarcascades/haarcascade_frontalface_alt.xml");

//    if (!)
//        return -1;

    cvtColor(frame, gray, CV_BGR2GRAY);
    equalizeHist(gray, gray);
    cascade.detectMultiScale(gray, faces, 1.2, 3);
    cout << faces.size() << 'f';

    return gray;
}

Mat binarize(const Mat& frame){

    Mat gray, kernel, hsv,skin;
    vector<Mat> rgb;
    GaussianBlur(frame,frame,Size(5,5),0);
    //split(frame,rgb);
    cvtColor(frame,gray,CV_BGR2GRAY);
    //rgb[1].copyTo(gray);
    threshold(gray,gray,75,255,THRESH_BINARY);

    kernel = getStructuringElement(MORPH_RECT,Size(5,5));
    dilate(gray,gray,kernel,Point(-1,-1),1);
    return gray;

//    cvtColor(frame,gray,CV_BGR2GRAY);
//    adaptiveThreshold(gray,
//                      gray,
//                      255,
//                      ADAPTIVE_THRESH_GAUSSIAN_C,
//                      CV_THRESH_BINARY,
//                      75,-10);

//    cvtColor(frame,hsv,CV_BGR2HSV);
//   // inRange(hsv, Scalar(6, 10, 60), Scalar(50, 150, 255), skin);
//    inRange(hsv, Scalar(0,51,89), Scalar(17,140,255), skin);
//    return skin;
}

bool is_good(vector<Point> contour){
// Eliminate too short or too long contours
// other constraints can be added here
    int cmin = 1000;
    int cmax = 30000;
    return (contour.size() > cmin && contour.size() < cmax);
}

void find_skin_blobs(const Mat& image, vector <vector<Point> >& skin_blobs){
    Mat binary_image;
    image.copyTo(binary_image);

    vector<vector<Point> > contours;
    findContours(binary_image,
                 contours, // a vector of contours
                 CV_RETR_EXTERNAL, // retrieve the external contours
                 CV_CHAIN_APPROX_NONE); // all pixels of each contours

    vector<vector<Point> > :: iterator itc = contours.begin();
    while(itc != contours.end()){
        //cout << contourArea(*itc) << endl;

        if(is_good(*itc)){
            skin_blobs.push_back(*itc);
        }
        ++itc;
    }
}


double dist(Blob b, Point p){

    Point center = b.get_blob_center();
    Size axes = b.get_blob_axes();

    double c = (p.x - center.x)/((double)axes.width);

    double d = (p.y - center.y)/((double)axes.height);

    double ad = b.get_angle();

    double ar = M_PI*ad/180.0; // change to radian


    double t1 = c*cos(ar) - d*sin(ar);
    double t1_2 = t1 * t1;

    double t2 = c*sin(ar) + d*cos(ar);
    double t2_2 = t2 * t2;

    return sqrt(t1_2 + t2_2);
}


void mark(Mat& image, const Point& p,const Scalar& color){
    int i = p.y;
    int j = p.x;
    image.at<Vec3b>(i,j)[0] = color[0];
    image.at<Vec3b>(i,j)[1] = color[1];
    image.at<Vec3b>(i,j)[2] = color[2];
}

Person apply_association_rules(Person hypothesis, Mat& image, Mat& frame){

    Blob h_face = hypothesis.get_face();
    Blob h_right_hand = hypothesis.get_right_hand();
    Blob h_left_hand = hypothesis.get_left_hand();

    h_face.draw(frame,true,Scalar(0,0,0));
    h_right_hand.draw(frame,true, Scalar(255,255,255));
    h_left_hand.draw(frame,true, Scalar(255,0,0));

    vector<Point> face_points;
    vector<Point> right_hand_points;
    vector<Point> left_hand_points;

    // to remember: write code to deal with joined hands that split
    // a hypothesis shared by more than one blob
    for(int i = 0; i < image.rows; i++)
        for(int j = 0; j < image.cols; j++){
            if(image.at<uchar>(i,j) == 255){
                Point p(j,i);
                int intensity = 255;
               // double alpha = 0.5;
                bool in_on = false;
                Scalar color = Scalar::all(0);
                double d1 = dist(h_face,p);
                if (d1 <= 1.0){
                    face_points.push_back(p);
                    in_on = true;
                    color[1] =  intensity;
                }

                double d2 = dist(h_right_hand,p);
                if (d2 <= 1.0){
                    right_hand_points.push_back(p);
                    in_on = true;
                    color[1] = color[2] = intensity;
                }

                double d3 = dist(h_left_hand,p);
                if (d3 <= 1.0){
                    left_hand_points.push_back(p);
                    in_on = true;
                    color[2] += intensity;
                }

                if (!in_on){
                    if (d1 < d2 &&  d1 < d3 && d1 < 2){
                        face_points.push_back(p);
                        color[1] += intensity;
                    }
                    if (d2 < d1 && d2 < d3 && d2 < 2){
                        right_hand_points.push_back(p);
                        color[1] += intensity;
                        color[2] += intensity;
                    }
                    if (d3 < d1 && d3 < d2 && d3 < 2){
                        left_hand_points.push_back(p);
                        color[2] += intensity;
                    }
                //    else{}
                }
                mark(frame,p,color);

            }
        }

    Blob face(h_face);
    Blob left_hand(h_left_hand);
    Blob right_hand(h_right_hand);

    if(face_points.size() > 5){
        RotatedRect rect_face = fitEllipse(Mat(face_points));
        Blob f(rect_face,face_points,Scalar(0,255,0));
        face = f;
    }
    if(right_hand_points.size() > 5){
        RotatedRect rect_right_hand = fitEllipse(Mat(right_hand_points));
        Blob rh(rect_right_hand, right_hand_points,Scalar(0,255,255));
        right_hand = rh;
    }
    if(left_hand_points.size() > 5){
        RotatedRect rect_left_hand = fitEllipse(Mat(left_hand_points));
        Blob lh(rect_left_hand, left_hand_points,Scalar(0,0,255));
        left_hand = lh;
    }

    Person p(face,right_hand,left_hand);

    return p;
}

bool nearer_circle(Point2f c1, Point2f c2){
    return (c1.x * c1.x + c1.y* c1.y) < (c2.x * c2.x + c2.y* c2.y);
}

Person get_init_person(vector<vector<Point> >& skin_blobs,const Mat& frame){
    vector<float> radii;
    vector<Point2f> centers;
    float radius;
    Point2f center;
    if(skin_blobs.size() == 3){
        for(int i = 0; i < skin_blobs.size(); i++){
            minEnclosingCircle(Mat(skin_blobs[i]),center,radius);
            radii.push_back(radius);
            centers.push_back(center);
        }
        sort(centers.begin(),centers.end(),nearer_circle);
        sort(radii.begin(),radii.end());
        int r = static_cast<int> (radii[0]/2);
        float a = 0.0;
        Point f = Point((int)centers[0].x,(int)centers[0].x);
        Blob face(f,Size(r,r),a,Scalar::all(0));

        Point rh = Point((int)centers[1].x,(int)centers[1].x);
        Blob rhand(rh,Size(r,r),a,Scalar::all(255));

        Point lh = Point((int)centers[2].x,(int)centers[2].x);

        Blob lhand(lh,Size(r,r),a,Scalar(255,0,0));
        Person p1(face,rhand,lhand);
        return p1;
    }
    else if(skin_blobs.size() == 2){

        for(int i = 0; i < skin_blobs.size(); i++){
            minEnclosingCircle(Mat(skin_blobs[i]),center,radius);
            radii.push_back(radius);
            centers.push_back(center);
        }
        sort(centers.begin(),centers.end(),nearer_circle);
        sort(radii.begin(),radii.end());
        int r = static_cast<int> (radii[0]/2);
        float a = 0.0;
        Point f_center = Point((int)centers[0].x,(int)centers[0].y);
        Point h_center = Point((int)centers[1].x,(int)centers[1].y);
        Blob face(f_center,Size(r,r),a,Scalar::all(0));
        Blob rhand(h_center,Size(r,r),a,Scalar::all(255));
        Blob lhand(h_center + Point(30,0),Size(r,r),a,Scalar(255,0,0));
        Person p1(face,rhand,lhand);
        return p1;
    }
    else{
        float a = 0.0;
        int r = frame.cols*.10;
        Point f_center = Point(frame.cols/2,frame.rows/3);
        Point rh_center = Point(frame.cols/2,frame.rows*4/5);
        Point lh_center = Point(frame.cols/2,frame.rows*4/5);
        Blob face(f_center,Size(r,r),a,Scalar::all(0));
        Blob rhand(rh_center,Size(r,r),a,Scalar::all(255));
        Blob lhand(lh_center + Point(30,0),Size(r,r),a,Scalar(255,0,0));
        Person p1(face,rhand,lhand);
        return p1;
    }

}



// A view requests from the model
// the information that it needs to generate an output representation
// mvc
// model - data, representation, object
// view - presentation, display, write
// controller - get input from user/file, talk to model, talk to view
int main(int argc, char* argv[]){
    Mat frame, image, blobs;
    VideoCapture cap;
    string win_name = "Tracking";

    //string file = "/Users/bingeb/data/bsl_selected/Dicta-Sign_BSL_S1-T1-a-AFR.mp4";

    string file = "/Users/bingeb/clara/data/SSL_JM_poem_cayak.mpg";
    if(!cap.open(file))
            return -1;

    //initialization for human body parts
    Person person, measured;

    char key;
    Person p_init;
    while(true){
        cap >> frame;
        Mat bin = binarize(frame);
        vector<vector<Point> > skin_blobs;
        find_skin_blobs(bin,skin_blobs);

        p_init = get_init_person(skin_blobs, frame);

        p_init.draw(bin,true);
        imshow(win_name,bin);
        key = waitKey();
        if((int)key == 27)
            break;
    }

    Tracker tracker(p_init);

    namedWindow(win_name);
    bool with_ellipse = true;
    int i = 0;
    //char key;
    while(true){
        cout << i++ << '\t';
        cap >> frame;
        frame.copyTo(image);
        blur(frame,frame,Size(5,5));

        if(frame.empty())
            break;
        key = waitKey(10);
        if((int)key == 27)
            break;
        if(key == ' ')
            waitKey();

        tracker.predict();

        person = tracker.get_person();
        // gets blobs or contours
        // blobs = get_blobs(frame,hist); // gets blobs from the next frame
        blobs = binarize(frame);
        //imshow(win_name,blobs);

        //  associates prediction with measurement
        measured = apply_association_rules(person, blobs,frame);
        measured.draw(frame, with_ellipse);

        tracker.correct(measured);
        tracker.print();

        imshow(win_name,frame);

        cout << endl;
    }
}


