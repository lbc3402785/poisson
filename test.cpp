#include "test.h"
#include "poisson.h"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "poisson.h"
using namespace Eigen;
Test::Test()
{

}
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatF;

void Test::testVectorlize()
{
    cv::Mat data=cv::Mat::ones(2,3,CV_32FC3);
    data.at<cv::Vec3f>(0,0)[0]=100;
    data.at<cv::Vec3f>(0,0)[1]=9;
    data.at<cv::Vec3f>(1,0)[2]=50;
    data.at<cv::Vec3f>(0,2)[1]=3;
    std::cout<<data<<std::endl;
    std::cout<<"-------"<<std::endl;
    std::cout<<data.size()<<std::endl;
    cv::Mat data1=data.reshape(0,2*3);
    std::cout<<data1<<std::endl;
    std::cout<<"-------"<<std::endl;
    std::cout<<data1.at<cv::Vec3f>(3)<<std::endl;
}

void Test::testCV2Eigen()
{
//    cv::Mat data=cv::Mat::zeros(3,5,CV_32FC1);
//    data.at<float>(0,0)=161;
//    data.at<float>(0,2)=120;
//    data.at<float>(1,4)=100;
//    Eigen::MatrixXf e;
//    cv::cv2eigen(data,e);
//    std::cout<<e<<std::endl;
//    std::cout<<"-------"<<std::endl;


//    cv::Mat data1=cv::Mat::zeros(3,5,CV_8S);
//    data1.at<short>(0,0)=1;
//    data1.at<short>(0,2)=22;
//    data1.at<short>(1,4)=41;
//    std::cout<<data1<<std::endl;
//    std::cout<<"-------"<<std::endl;
//    Eigen::MatrixXi e1;
//    cv::cv2eigen(data1,e1);
//    std::cout<<e1<<std::endl;
//    std::cout<<"-------"<<std::endl;

//    cv::Mat data2=cv::Mat::zeros(3,5,CV_8U);
//    data2.at<ushort>(0,0)=1;
//    data2.at<ushort>(0,2)=22;
//    data2.at<ushort>(1,4)=41;
//    std::cout<<data2<<std::endl;
//    std::cout<<"-------"<<std::endl;
//    Eigen::MatrixXi e2;
//    cv::cv2eigen(data2,e2);
//    std::cout<<e2<<std::endl;
//    std::cout<<"-------"<<std::endl;

//    cv::Mat data3=cv::Mat::zeros(3,5,CV_32FC1);
//    data3.at<float>(0,0)=1.99;
//    data3.at<float>(0,2)=22;
//    data3.at<float>(1,4)=41.3;
//    std::cout<<data3<<std::endl;
//    std::cout<<"-------"<<std::endl;
//    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> e3;
//    cv::cv2eigen(data3,e3);
//    std::cout<<e3<<std::endl;
//    std::cout<<"-------"<<std::endl;
//    cv::Mat data4;
//    cv::eigen2cv(e3,data4);
//    std::cout<<data4<<std::endl;
//    std::cout<<"-------"<<std::endl;
//    std::cout<<(data4.type()==CV_32FC1)<<std::endl;
//    std::cout<<"-------"<<std::endl;
//    std::cout<<(data4.channels()==1)<<std::endl;
//    std::cout<<"-------"<<std::endl;

    cv::Mat data5=cv::Mat::zeros(3,5,CV_32FC1);
    data5.at<float>(0,0)=1.99;
    data5.at<float>(0,2)=22;
    data5.at<float>(1,4)=41.3;
    std::cout<<data5<<std::endl;
    std::cout<<"-------"<<std::endl;
    cv::Mat data6=data5.reshape(0,3*5);
    Eigen::MatrixXf e4;
    cv::cv2eigen(data6,e4);
    std::cout<<data6<<std::endl;
    std::cout<<"-------"<<std::endl;
    Eigen::VectorXf t=e4;
    cv::Mat data7;
    cv::eigen2cv(t,data7);
    cv::Mat data8=data7.reshape(0,3);
    std::cout<<data8<<std::endl;
    std::cout<<"-------"<<std::endl;
}

void Test::testMaskBound()
{
    cv::Mat mask0=cv::imread("data\\M.png",0);
    cv::Mat mask=(mask0==255);
    cv::Mat bound=getMaskBound(mask);
    cv::imshow("mask",mask);
    cv::imshow("bound",bound*255);
    cv::waitKey();
}

void Test::testPoisson()
{
    cv::Mat mask0=cv::imread("data\\M.png",0);

    cv::Mat src=cv::imread("data\\orig.png");
    cv::Mat dst=cv::imread("data\\T.png");
    cv::resize(src,src,cv::Size(),0.5,0.5);
    cv::resize(dst,dst,cv::Size(),0.5,0.5);
    cv::resize(mask0,mask0,cv::Size(),0.5,0.5);
    //-------------------------------------
    cv::resize(dst,dst,cv::Size(),0.5,0.5);
    cv::Mat mask=(mask0==255);
    poisson(src,dst,mask);

}

void Test::testLapician()
{
     cv::Mat src=cv::imread("data\\orig.png");
     cv::resize(src,src,cv::Size(),0.5,0.5);


     cv::Mat blueSrc,greenSrc,redSrc;
     cv::Mat src3[3];
     //------------------------
     cv::split(src,src3);
     cv::imshow("src3[1]",src3[1]);
     cv::waitKey();
     Eigen::SparseMatrix<float> L=createLapician(src.rows,src.cols);//column vector of lapician
      src3[1].convertTo(src3[1],CV_32FC1,1);
     Eigen::MatrixXf M;
     cv::cv2eigen(src3[1],M);
     Eigen::MatrixXf V=Eigen::Map<MatrixXf> (M.data(),M.size(),1);

     Eigen::MatrixXf A=L*V;
     Eigen::MatrixXf D=Eigen::Map<MatrixXf> (A.data(),M.rows(),M.cols());

     cv::Mat result;
     cv::eigen2cv(D,result);
     cv::imshow("result",result);
     cv::waitKey();
}

void Test::testReshape()
{

    cv::Mat data5=cv::Mat::zeros(2,4,CV_32FC1);
    data5.at<float>(0,0)=1.99;
    data5.at<float>(0,1)=22;
    data5.at<float>(0,2)=32;
    data5.at<float>(1,2)=22;
    data5.at<float>(1,3)=41.3;
    std::cout<<data5<<std::endl;
    std::cout<<"-------"<<std::endl;

//     cv::Mat data55=data5.reshape(0,2*4);
//     std::cout<<data55<<std::endl;
//     std::cout<<"-------"<<std::endl;

//     cv::Mat data44=data55.reshape(0,2);
//     std::cout<<data44<<std::endl;
//     std::cout<<"-------"<<std::endl;

//     cv::Mat data555=data5.reshape(0,1);
//     std::cout<<data555<<std::endl;
//     std::cout<<"-------"<<std::endl;

//     cv::Mat data444=data5.reshape(0,2);
//     std::cout<<data444<<std::endl;
//     std::cout<<"-------"<<std::endl;

    Eigen::MatrixXf e4;
    cv::cv2eigen(data5,e4);
//    std::cout<<e4<<std::endl;
//    std::cout<<"-------"<<std::endl;
    Eigen::MatrixXf e5=Eigen::Map<MatrixXf> (e4.data(), 2*4,1);
    std::cout<<e5<<std::endl;
    std::cout<<"-------"<<std::endl;
    cv::Mat data6;
    cv::eigen2cv(e5,data6);
    std::cout<<data6<<std::endl;
    std::cout<<"-------"<<std::endl;
    cv::Mat data7=data6.reshape(0,2);
    std::cout<<data7<<std::endl;
    std::cout<<"-------"<<std::endl;

    Eigen::MatrixXf e44;
    cv::cv2eigen(data5,e44);
//    std::cout<<e44<<std::endl;
//    std::cout<<"-------"<<std::endl;
    Eigen::MatrixXf e55=Eigen::Map<MatrixXf> (e44.data(),1,2*4);
    std::cout<<e55<<std::endl;
    std::cout<<"-------"<<std::endl;
    cv::Mat data66;
    cv::eigen2cv(e55,data66);
    std::cout<<data66<<std::endl;
    std::cout<<"-------"<<std::endl;
    cv::Mat data77=data66.reshape(0,2);
    std::cout<<data77<<std::endl;
    std::cout<<"-------"<<std::endl;
}

void Test::testReshape2()
{
    cv::Mat data5=cv::Mat::zeros(2,4,CV_32FC1);
    data5.at<float>(0,0)=1.99;
    data5.at<float>(0,1)=22;
    data5.at<float>(0,2)=32;
    data5.at<float>(1,0)=87;
    data5.at<float>(1,3)=41.3;
    std::cout<<data5<<std::endl;
    std::cout<<"-------"<<std::endl;
    Eigen::MatrixXf e0;
    cv::cv2eigen(data5,e0);
    std::cout<<e0<<std::endl;
    std::cout<<"-------"<<std::endl;

    cv::Mat data55=data5.reshape(0,2*4);
//    std::cout<<data55<<std::endl;
//    std::cout<<"-------"<<std::endl;

    cv::Mat data66=data5.reshape(0,1);
//    std::cout<<data66<<std::endl;
//    std::cout<<"-------"<<std::endl;
    std::cout<<"++++++++++++"<<std::endl;

    Eigen::MatrixXf e4;
    cv::cv2eigen(data55,e4);
//    std::cout<<e4<<std::endl;
//    std::cout<<"-------"<<std::endl;
    Eigen::MatrixXf e5=Eigen::Map<MatrixXf> (e4.data(), 2,4);
    std::cout<<e5<<std::endl;
    std::cout<<"-------"<<std::endl;

    Eigen::MatrixXf e6;
    cv::cv2eigen(data66,e6);
//    std::cout<<e6<<std::endl;
//    std::cout<<"-------"<<std::endl;
    Eigen::MatrixXf e7=Eigen::Map<MatrixXf> (e6.data(), 2,4);
    std::cout<<e7<<std::endl;
    std::cout<<"-------"<<std::endl;
}

void Test::testReshape3()
{
    cv::Mat data5=cv::Mat::zeros(2,4,CV_32FC1);
    data5.at<float>(0,0)=1.99;
    data5.at<float>(0,1)=22;
    data5.at<float>(0,2)=32;
    data5.at<float>(1,0)=87;
    data5.at<float>(1,3)=41.3;
    std::cout<<data5<<std::endl;
    std::cout<<"-------"<<std::endl;

    Eigen::MatrixXf e0;
    cv::cv2eigen(data5,e0);
    std::cout<<e0<<std::endl;
    std::cout<<"-------"<<std::endl;

    Eigen::MatrixXf e1=Eigen::Map<MatrixXf> (e0.data(), 2*4,1);

    std::cout<<e1<<std::endl;
    std::cout<<"-------"<<std::endl;

//    Eigen::MatrixXf e2=Eigen::Map<MatrixXf> (e1.data(), 2,4);

//    std::cout<<e2<<std::endl;
//    std::cout<<"-------"<<std::endl;
    cv::Mat data6;
    cv::eigen2cv(e1,data6);
    std::cout<<data6<<std::endl;
    std::cout<<"-------"<<std::endl;

    cv::Mat data7;
    data7=data6.reshape(0,2);
    std::cout<<data7<<std::endl;
    std::cout<<"-------"<<std::endl;
}

void Test::testReshape4()
{
    cv::Mat data5=cv::Mat::zeros(2,4,CV_32FC1);
    data5.at<float>(0,0)=1.99;
    data5.at<float>(0,1)=22;
    data5.at<float>(0,2)=32;
    data5.at<float>(1,0)=87;
    data5.at<float>(1,3)=41.3;
    std::cout<<data5<<std::endl;
    std::cout<<"-------"<<std::endl;

    Eigen::MatrixXf e0;
    cv::cv2eigen(data5,e0);
    std::cout<<e0<<std::endl;
    std::cout<<"-------"<<std::endl;

    Eigen::MatrixXf e1=Eigen::Map<MatrixXf> (e0.data(), 2*4,1);

    std::cout<<e1<<std::endl;
    std::cout<<"-------"<<std::endl;


    Eigen::MatrixXf e2=Eigen::Map<MatrixXf> (e1.data(), 2,4);
//    Eigen::MatrixXf e2=Eigen::Map<MatrixXf> (e1.data(), 2,4);

//    std::cout<<e2<<std::endl;
//    std::cout<<"-------"<<std::endl;
    cv::Mat data6;
    cv::eigen2cv(e2,data6);
    std::cout<<data6<<std::endl;
    std::cout<<"-------"<<std::endl;
}
cv::Mat replaceByMask(cv::Mat src,cv::Mat dst,cv::Mat mask){
    Eigen::MatrixXf M1;
    Eigen::MatrixXf M2;
    cv::cv2eigen(src,M1);
    cv::cv2eigen(dst,M2);
    Eigen::MatrixXf V=Eigen::Map<MatrixXf> (M1.data(), M1.rows()*M1.cols(),1);//column vectorization of Src
    Eigen::MatrixXf I=Eigen::Map<MatrixXf> (M2.data(), M1.rows()*M1.cols(),1);//column vectorization of Dst
    cv::Mat msk=mask.clone();
    msk/=255;
    msk.convertTo(msk,CV_32FC1,1);
    Eigen::SparseMatrix<float> POmega=createProjection(msk);
    Eigen::MatrixXf Omega=POmega*V;
    for(int j=0;j<msk.cols;j++){
        for(int i=0;i<msk.rows;i++){
            if(msk.at<float>(i,j)>0)
            {
                int idx=j*msk.rows+i;
                I(idx)=V(idx);
            }
        }
    }
    Eigen::MatrixXf D=Eigen::Map<MatrixXf> (I.data(),M2.rows(),M2.cols());
    cv::Mat result;
    cv::eigen2cv(D,result);
    return result;
}
cv::Mat replaceByMask1(cv::Mat src,cv::Mat dst,cv::Mat mask){
    Eigen::MatrixXf M1;
    Eigen::MatrixXf M2;
    cv::cv2eigen(src,M1);
    cv::cv2eigen(dst,M2);
    Eigen::MatrixXf V=Eigen::Map<MatrixXf> (M1.data(), M1.rows()*M1.cols(),1);//column vectorization of Src
    Eigen::MatrixXf I=Eigen::Map<MatrixXf> (M2.data(), M1.rows()*M1.cols(),1);//column vectorization of Dst
    cv::Mat msk=mask.clone();
    msk/=255;
    msk.convertTo(msk,CV_32FC1,1);
//    Eigen::SparseMatrix<float> POmega=createProjection(msk);
    int n=cv::countNonZero(msk);
    std::cout<<"n:"<<n<<std::endl;
    Eigen::SparseMatrix<float> SOmega=createBinary(n,msk);
    Eigen::MatrixXf Omega=SOmega*V;
    //std::cout<<"Omega:"<<Omega<<std::endl;
    std::cout<<"-------------------------------------"<<std::endl;
    int count=0;
    for(int j=0;j<msk.cols;j++){
        for(int i=0;i<msk.rows;i++){
            if(msk.at<float>(i,j)>0)
            {
                int idx=j*msk.rows+i;
//                std::cout<<"I(idx):"<<I(idx)<<std::endl;
//                std::cout<<"Omega(count):"<<Omega(count)<<std::endl;
                I(idx)=Omega(count);
                count++;
            }
        }
    }
    std::cout<<"count:"<<count<<std::endl;
    Eigen::MatrixXf D=Eigen::Map<MatrixXf> (I.data(),M2.rows(),M2.cols());
    cv::Mat result;
    cv::eigen2cv(D,result);
    return result;
}
void Test::testMask(){
    cv::Mat mask0=cv::imread("data\\M.png",0);
    cv::Mat mask=(mask0==255);
    cv::resize(mask,mask,cv::Size(),0.5,0.5);
    cv::Mat src=cv::imread("data\\orig.png");
    cv::resize(src,src,cv::Size(),0.5,0.5);


    cv::Mat blueSrc,greenSrc,redSrc;
    cv::Mat src3[3];

    cv::split(src,src3);
    src3[0].convertTo(blueSrc, CV_32FC1);
    src3[1].convertTo(greenSrc, CV_32FC1);
    src3[2].convertTo(redSrc, CV_32FC1);


    cv::Mat dst=cv::imread("data\\T.png");
    cv::resize(dst,dst,cv::Size(),0.25,0.25);


    cv::Mat blueDst,greenDst,redDst;
    cv::Mat dst3[3];
    cv::split(dst,dst3);
    dst3[0].convertTo(blueDst, CV_32FC1);
    dst3[1].convertTo(greenDst, CV_32FC1);
    dst3[2].convertTo(redDst, CV_32FC1);

    cv::Mat boundMask=getMaskBound(mask);
    boundMask.convertTo(boundMask,CV_32FC1,1);
    cv::Mat b=replaceByMask1(blueSrc,blueDst,mask)/*blueDst.mul(boundMask)*/;
    cv::Mat g=replaceByMask1(greenSrc,greenDst,mask)/*greenDst.mul(boundMask)*/;
    cv::Mat r=replaceByMask1(redSrc,redDst,mask)/*redDst.mul(boundMask)*/;
    cv::Mat bgr[3];
    b.convertTo(bgr[0], CV_8UC1);
    g.convertTo(bgr[1], CV_8UC1);
    r.convertTo(bgr[2], CV_8UC1);
    merge(bgr,3, dst);

    cv::imshow("dst",dst);
    cv::waitKey();
}
