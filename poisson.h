#ifndef POISSON_H
#define POISSON_H
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/core/eigen.hpp>
#include "lineargradient.h"
#include "preconditionedconjugategradient.h"
cv::Mat getMaskBound(cv::Mat msk){
    cv::Mat bound=cv::Mat::zeros(msk.size(),msk.type());
    for(int i=0;i<msk.rows;i++){
        for(int j=0;j<msk.cols;j++){
            if(msk.at<uchar>(i,j)>0){
                if(i>0&&i<msk.rows-1&&j>0&&j<msk.cols-1){
                    int interior=0;
                    if(msk.at<uchar>(i,j-1)>0){
                        interior++;
                    }
                    if(msk.at<uchar>(i,j+1)>0){
                        interior++;
                    }
                    if(msk.at<uchar>(i-1,j)>0){
                        interior++;
                    }
                    if(msk.at<uchar>(i+1,j)>0){
                        interior++;
                    }
                    if(interior<=3){
                        bound.at<uchar>(i,j)=1;
                    }
                }else{
                    bound.at<uchar>(i,j)=1;
                }
            }
        }
    }
    return  bound;
}
//cv::Mat getImageBound(cv::Mat image){
//    int rows=image.rows;
//    int cols=image.cols;
//    cv::Mat bound=cv::Mat::zeros(rows,cols,CV_8U);
//    for(int i=0;i<rows;i++){
//        bound
//    }
//}
//Eigen::SparseMatrix<float> createLapician(int rows,int cols){
//    int N=rows*cols;
//    Eigen::SparseMatrix<float> L;
//    L.resize(N,N);
//    std::vector<Eigen::Triplet<float>> trips;
//    for(int i=0;i<rows;i++){
//        for(int j=0;j<cols;j++){
//            int row=i*cols+j;
//            int col=j*rows+i;
//            if(i==0){
//                if(j==0){
//                    trips.emplace_back(Eigen::Triplet<float>(row,col,-2));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col+1,1));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col+rows,1));
//                }else if(j==cols-1){
//                    trips.emplace_back(Eigen::Triplet<float>(row,col,-2));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col+1,1));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col-rows,1));
//                }else{
//                    trips.emplace_back(Eigen::Triplet<float>(row,col,-3));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col+1,1));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col+rows,1));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col-rows,1));
//                }
//            }else if(i==rows-1){
//                if(j==0){
//                    trips.emplace_back(Eigen::Triplet<float>(row,col,-2));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col-1,1));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col+rows,1));
//                }else if(j==cols-1){
//                    trips.emplace_back(Eigen::Triplet<float>(row,col,-2));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col-1,1));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col-rows,1));
//                }else{
//                    trips.emplace_back(Eigen::Triplet<float>(row,col,-3));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col-1,1));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col+rows,1));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col-rows,1));
//                }
//            }else{
//                if(j==0){
//                    trips.emplace_back(Eigen::Triplet<float>(row,col,-3));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col-1,1));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col+1,1));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col+rows,1));
//                }else if(j==cols-1){
//                    trips.emplace_back(Eigen::Triplet<float>(row,col,-3));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col-1,1));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col+1,1));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col-rows,1));
//                }else{
//                    trips.emplace_back(Eigen::Triplet<float>(row,col,-4));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col-1,1));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col+1,1));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col-rows,1));
//                    trips.emplace_back(Eigen::Triplet<float>(row,col+rows,1));
//                }
//            }
//        }
//    }
//    L.setFromTriplets(trips.begin(),trips.end());
//    return L;
//}
Eigen::SparseMatrix<float> createDx(int rows,int cols){
    int N=rows*cols;
    Eigen::SparseMatrix<float> L;
    L.resize(N,N);
    std::vector<Eigen::Triplet<float>> trips;
    for(int j=0;j<cols;j++){
        for(int i=0;i<rows;i++){
            int row=j*rows+i;
            int col=row;
            if(j==0){
                trips.emplace_back(Eigen::Triplet<float>(row,col,-1));
                trips.emplace_back(Eigen::Triplet<float>(row,col+rows,1));
            }else if(j==cols-1){
                trips.emplace_back(Eigen::Triplet<float>(row,col,1));
                trips.emplace_back(Eigen::Triplet<float>(row,col-rows,-1));
            }else{
                trips.emplace_back(Eigen::Triplet<float>(row,col+rows,1));
                trips.emplace_back(Eigen::Triplet<float>(row,col-rows,-1));
            }
        }
    }
}
Eigen::SparseMatrix<float> createDy(int rows,int cols){
    int N=rows*cols;
    Eigen::SparseMatrix<float> L;
    L.resize(N,N);
    std::vector<Eigen::Triplet<float>> trips;
    for(int j=0;j<cols;j++){
        for(int i=0;i<rows;i++){
            int row=j*rows+i;
            int col=row;
            if(i==0){
                trips.emplace_back(Eigen::Triplet<float>(row,col,-1));
                trips.emplace_back(Eigen::Triplet<float>(row,col+1,1));
            }else if(i==rows-1){
                trips.emplace_back(Eigen::Triplet<float>(row,col,1));
                trips.emplace_back(Eigen::Triplet<float>(row,col-1,-1));
            }else{
                trips.emplace_back(Eigen::Triplet<float>(row,col+1,1));
                trips.emplace_back(Eigen::Triplet<float>(row,col-1,-1));
            }
        }
    }
}

Eigen::SparseMatrix<float> createLapician(int rows,int cols){
    int N=rows*cols;
    Eigen::SparseMatrix<float> L;
    L.resize(N,N);
    std::vector<Eigen::Triplet<float>> trips;
    for(int j=0;j<cols;j++){
        for(int i=0;i<rows;i++){
            int row=j*rows+i;
            int col=row;
            if(i==0){
                if(j==0){
                    trips.emplace_back(Eigen::Triplet<float>(row,col,-2));
                    trips.emplace_back(Eigen::Triplet<float>(row,col+1,1));
                    trips.emplace_back(Eigen::Triplet<float>(row,col+rows,1));
                }else if(j==cols-1){
                    trips.emplace_back(Eigen::Triplet<float>(row,col,-2));
                    trips.emplace_back(Eigen::Triplet<float>(row,col+1,1));
                    trips.emplace_back(Eigen::Triplet<float>(row,col-rows,1));
                }else{
                    trips.emplace_back(Eigen::Triplet<float>(row,col,-3));
                    trips.emplace_back(Eigen::Triplet<float>(row,col+1,1));
                    trips.emplace_back(Eigen::Triplet<float>(row,col+rows,1));
                    trips.emplace_back(Eigen::Triplet<float>(row,col-rows,1));
                }
            }else if(i==rows-1){
                if(j==0){
                    trips.emplace_back(Eigen::Triplet<float>(row,col,-2));
                    trips.emplace_back(Eigen::Triplet<float>(row,col-1,1));
                    trips.emplace_back(Eigen::Triplet<float>(row,col+rows,1));
                }else if(j==cols-1){
                    trips.emplace_back(Eigen::Triplet<float>(row,col,-2));
                    trips.emplace_back(Eigen::Triplet<float>(row,col-1,1));
                    trips.emplace_back(Eigen::Triplet<float>(row,col-rows,1));
                }else{
                    trips.emplace_back(Eigen::Triplet<float>(row,col,-3));
                    trips.emplace_back(Eigen::Triplet<float>(row,col-1,1));
                    trips.emplace_back(Eigen::Triplet<float>(row,col+rows,1));
                    trips.emplace_back(Eigen::Triplet<float>(row,col-rows,1));
                }
            }else{
                if(j==0){
                    trips.emplace_back(Eigen::Triplet<float>(row,col,-3));
                    trips.emplace_back(Eigen::Triplet<float>(row,col-1,1));
                    trips.emplace_back(Eigen::Triplet<float>(row,col+1,1));
                    trips.emplace_back(Eigen::Triplet<float>(row,col+rows,1));
                }else if(j==cols-1){
                    trips.emplace_back(Eigen::Triplet<float>(row,col,-3));
                    trips.emplace_back(Eigen::Triplet<float>(row,col-1,1));
                    trips.emplace_back(Eigen::Triplet<float>(row,col+1,1));
                    trips.emplace_back(Eigen::Triplet<float>(row,col-rows,1));
                }else{
                    trips.emplace_back(Eigen::Triplet<float>(row,col,-4));
                    trips.emplace_back(Eigen::Triplet<float>(row,col-1,1));
                    trips.emplace_back(Eigen::Triplet<float>(row,col+1,1));
                    trips.emplace_back(Eigen::Triplet<float>(row,col-rows,1));
                    trips.emplace_back(Eigen::Triplet<float>(row,col+rows,1));
                }
            }
        }
    }
    L.setFromTriplets(trips.begin(),trips.end());
    return L;
}
Eigen::SparseMatrix<float> createProjection(cv::Mat mask){
    int N=mask.rows*mask.cols;
    Eigen::SparseMatrix<float> P;
    P.resize(N,N);
    Eigen::MatrixXf M;
    cv::cv2eigen(mask,M);
    Eigen::MatrixXf V=Eigen::Map<Eigen::MatrixXf> (M.data(),M.size(),1);//column vector of mask
    std::vector<Eigen::Triplet<float>> trips;
    for(int i=0;i<N;i++){
        trips.emplace_back(i,i,V(i));
    }
    P.setFromTriplets(trips.begin(),trips.end());
    return P;
}



Eigen::SparseMatrix<float> createBinary(int n,cv::Mat mask){
    int N=mask.rows*mask.cols;
    Eigen::SparseMatrix<float> S;
    S.resize(n,N);
    std::vector<Eigen::Triplet<float>> trips;
    int count=0;
    for(int j=0;j<mask.cols;j++){
        for(int i=0;i<mask.rows;i++){
            if(mask.at<float>(i,j)>0)
            {
                int idx=j*mask.rows+i;
                trips.emplace_back(count,idx,1);
                count++;
            }
        }
    }
    S.setFromTriplets(trips.begin(),trips.end());
    return S;
}
cv::Mat poissonSingleChannel(cv::Mat src,cv::Mat dst,cv::Mat msk,Eigen::SparseMatrix<float> L,Eigen::SparseMatrix<float> POmega,
                          Eigen::SparseMatrix<float> PAlphaOmega,Eigen::SparseMatrix<float> SOmega){

    using namespace Eigen;
    Eigen::MatrixXf M1;
    Eigen::MatrixXf M2;
    cv::cv2eigen(src,M1);
    cv::cv2eigen(dst,M2);
    Eigen::MatrixXf V=Eigen::Map<MatrixXf> (M1.data(), M1.rows()*M1.cols(),1);//column vectorization of Src
    Eigen::MatrixXf I=Eigen::Map<MatrixXf> (M2.data(), M1.rows()*M1.cols(),1);//column vectorization of Dst

    Eigen::SparseMatrix<float> A=SOmega*L*POmega*SOmega.transpose();
//    std::cout<<"A:"<<A.rows()<<","<<A.cols()<<std::endl;
    Eigen::MatrixXf b=SOmega*L*V-(SOmega*L*PAlphaOmega)*I;
//    std::cout<<"b:"<<b<<std::endl;
//    Eigen::VectorXf X = A.colPivHouseholderQr().solve(b);//nx1

    Eigen::SparseMatrix<float> ATA=A.transpose()*A;
    Eigen::SparseMatrix<float> Id(ATA.rows(),ATA.rows());
    Id.setIdentity();
    ATA+=1e-5*Id;
    Eigen::Diagonal<Eigen::SparseMatrix<float>> d=ATA.diagonal();
    Eigen::SparseMatrix<float> P(d.rows(),d.cols());
     std::vector<Eigen::Triplet<float>> trips;
     int count=0;
    for(int i=0;i<d.rows();i++){
        if(std::fabs(d(i))>FLT_EPSILON){
            trips.emplace_back(i,i,1/d(i));
            count++;
        }
    }
    P.resize(count,count);
    P.setFromTriplets(trips.begin(),trips.end());
    Eigen::VectorXf bt=A.transpose()*b;
    Eigen::VectorXf X1(bt.rows());

    PreConditionedConjugateGradient<Eigen::SparseMatrix<float>,Eigen::VectorXf,float> pgc;
    PreConditionedConjugateGradient<Eigen::SparseMatrix<float>,Eigen::VectorXf,float>::Status status=pgc.solve(ATA,bt,X1,&P);
    std::cout<<status.info<<std::endl;
    std::cout<<status.numIterations<<std::endl;
   std::cout<<"444"<<std::endl;
   count=0;
    for(int j=0;j<msk.cols;j++){
        for(int i=0;i<msk.rows;i++){
            if(msk.at<float>(i,j)>0)
            {
                int idx=j*msk.rows+i;
//                I(idx)=X(count++);
                I(idx)=X1(count++);
            }
        }
    }
    std::cout<<"count:"<<count<<std::endl;
    Eigen::MatrixXf D=Eigen::Map<MatrixXf> (I.data(),M2.rows(),M2.cols());
    cv::Mat result;
    cv::eigen2cv(D,result);
    return result;
}
void poisson(cv::Mat src,cv::Mat dst,cv::Mat mask){

    assert(src.size()==dst.size());
    cv::Mat blueSrc,greenSrc,redSrc;
    cv::Mat blueDst,greenDst,redDst;
    cv::Mat src3[3];
    cv::Mat dst3[3];
    //------------------------
    cv::split(src,src3);
    src3[0].convertTo(blueSrc, CV_32FC1);
    src3[1].convertTo(greenSrc, CV_32FC1);
    src3[2].convertTo(redSrc, CV_32FC1);
    //------------------------
    cv::split(dst,dst3);
    dst3[0].convertTo(blueDst, CV_32FC1);
    dst3[1].convertTo(greenDst, CV_32FC1);
    dst3[2].convertTo(redDst, CV_32FC1);
    //------------------------

    cv::Mat msk=mask.clone();
    cv::Mat boundMask=getMaskBound(msk);
//    cv::imshow("boundMask",boundMask*255);
//    cv::waitKey();
    msk/=255;
    msk.convertTo(msk,CV_32FC1,1);
    Eigen::SparseMatrix<float> L=createLapician(dst.rows,dst.cols);//column vector of lapician

    Eigen::SparseMatrix<float> POmega=createProjection(msk);

    boundMask.convertTo(boundMask,CV_32FC1,1);
    Eigen::SparseMatrix<float> PAlphaOmega=createProjection(boundMask);

    int n=cv::countNonZero(msk);
    std::cout<<n<<std::endl;
    Eigen::SparseMatrix<float> SOmega=createBinary(n,msk);

    //------------------------
    cv::Mat b=poissonSingleChannel(blueSrc,blueDst,msk,L,POmega,POmega,SOmega);
    cv::Mat g=poissonSingleChannel(greenSrc,greenDst,msk,L,POmega,POmega,SOmega);
    cv::Mat r=poissonSingleChannel(redSrc,redDst,msk,L,POmega,POmega,SOmega);
    cv::Mat bgr[3];
    b.convertTo(bgr[0], CV_8UC1);
    g.convertTo(bgr[1], CV_8UC1);
    r.convertTo(bgr[2], CV_8UC1);
    merge(bgr,3, dst);
    cv::imshow("dst",dst);
    cv::waitKey();
}

#endif // POISSON_H
