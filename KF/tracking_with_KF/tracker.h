#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include "person.h"

using namespace cv;
using namespace std;

class Tracker
{
public:
    Tracker();
    Tracker(Person p1);
    void predict();
    void correct(Person& measured);
    Person get_person();
    void print();

private:
    Person _person;
};

#endif // TRACKER_H
