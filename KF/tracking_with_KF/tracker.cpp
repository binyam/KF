#include "tracker.h"
#include "iostream"

using namespace std;

Tracker::Tracker()
{
    Person p1;
    _person = p1;
}

Tracker::Tracker(Person p1){
    _person = p1;
}

void Tracker::predict(){
    _person.predict();
}

Person Tracker::get_person(){
    return _person;
}

void Tracker::print(){
    cout << _person.get_face().get_blob_center() << '\t';
    cout << _person.get_right_hand().get_blob_center() << '\t';
    cout << _person.get_left_hand().get_blob_center() << '\t';
}

void Tracker::correct(Person& measured){
    Mat_<float> face_measurement(2,1);
    Mat_<float> rhand_measurement(2,1);
    Mat_<float> lhand_measurement(2,1);

    face_measurement(0) = static_cast<float>(measured.get_face().get_blob_center().x);
    face_measurement(1) = static_cast<float>(measured.get_face().get_blob_center().y);

    rhand_measurement(0) = static_cast<float>(measured.get_right_hand().get_blob_center().x);
    rhand_measurement(1) = static_cast<float>(measured.get_right_hand().get_blob_center().y);

    lhand_measurement(0) = static_cast<float>(measured.get_left_hand().get_blob_center().x);
    lhand_measurement(1) = static_cast<float>(measured.get_left_hand().get_blob_center().y);

    _person.correct(face_measurement,
                   rhand_measurement,
                   lhand_measurement);

}
