#-------------------------------------------------
#
# Project created by QtCreator 2013-01-08T15:34:56
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = tracking_with_KalmanFilter
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    tracker.cpp \
    person.cpp


INCLUDEPATH += /Users/bingeb/opencv243/include
LIBS += -L/Users/bingeb/opencv243/lib \
-lopencv_core \
-lopencv_highgui \
-lopencv_imgproc \
-lopencv_features2d \
-lopencv_ml \
-lopencv_video \
-lopencv_objdetect

HEADERS += \
    blob.h \
    person.h \
    tracker.h











