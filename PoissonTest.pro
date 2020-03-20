QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS
OBJECTS_DIR=$${PWD}/build
QMAKE_CXXFLAGS+= /std:c++17
ZLIB_INCLUDE_DIRS="D:\soft\opencv3\sources\3rdparty\zlib"
OPENCV_INCLUDE_DIRS=D:\soft\opencv3\build\include
OPENCV_LIBRARY_DIRS=D:\soft\opencv3\build\x64\vc15\lib
# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
INCLUDEPATH+=$$OPENCV_INCLUDE_DIRS
INCLUDEPATH+=D:\soft\eigen-eigen-323c052e1731
LIBS+=-L$$OPENCV_LIBRARY_DIRS -lopencv_world344
SOURCES += \
        main.cpp \
        test.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    lineargradient.h \
    poisson.h \
    preconditionedconjugategradient.h \
    test.h
