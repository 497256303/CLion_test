#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cv.h>
using namespace cv;
using namespace std;

int gray_invariant_lbp(IplImage *src, int height, int width, int num_sp, CvPoint *spoint)
{
	IplImage *target,*hist;
	int i,j,k,box_x,box_y,orign_x,orign_y,dx,dy,tx,ty,fy,fx,cy,cx,v;
	double min_x,max_x,min_y,max_y,w1,w2,w3,w4,N,x,y;
	int *result;
	float dishu;

	dishu = 2.0;
	max_x=0;max_y=0;min_x=0;min_y=0;
	for (k=0;k<num_sp;k++)
	{
		if (max_x<spoint[k].x)
		{
			max_x=spoint[k].x;
		}
		if (max_y<spoint[k].y)
		{
			max_y=spoint[k].y;
		}
		if (min_x>spoint[k].x)
		{
			min_x=spoint[k].x;
		}
		if (min_y>spoint[k].y)
		{
			min_y=spoint[k].y;
		}
	}

	//计算模版大小
	box_x = ceil(MAX(max_x,0)) - floor(MIN(min_x,0)) + 1;
	box_y = ceil(MAX(max_y,0)) - floor(MIN(min_y,0)) + 1;

	if (width<box_x||height<box_y)
	{
		printf("Too small input image. Should be at least (2*radius+1) x (2*radius+1)");
		return -1;
	}

	//计算可滤波图像大小,opencv图像数组下标从0开始
	orign_x = 0 - floor(MIN(min_x,0));//起点
	orign_y = 0 - floor(MIN(min_x,0));

	dx = width - box_x+1;//终点
	dy = height - box_y+1;

	int cols = pow(dishu,(float)num_sp);
	hist = cvCreateImage(cvSize(300,200),IPL_DEPTH_8U,3);//直方图图像
	target = cvCreateImage(cvSize(dx,dy),IPL_DEPTH_8U,1);
	result = (int *)malloc(sizeof(int)*dx*dy);
	double *val_hist = (double *)malloc(sizeof(double)*cols);   //直方图数组

	memset(result,0,sizeof(int)*dx*dy);
	CvRect roi =cvRect(orign_x, orign_y, dx, dy);
	cvSetImageROI(src, roi);
	cvCopy(src, target);
	cvResetImageROI(src);
	cvSaveImage("/data/haha.jpg",src);

	for ( k = 0; k<num_sp;k++)
	{
		x = spoint[k].x+orign_x;
		y = spoint[k].y+orign_y;

		//二线性插值图像
		fy = floor(y);  //向下取整
		fx = floor(x);
		cy = ceil(y);   //向上取整
		cx = ceil(x);
		ty = y - fy;
		tx = x - fx;
		w1 = (1 - tx) * (1 - ty);
		w2 = tx  * (1 - ty);
		w3 = (1 - tx) * ty ;
		w4 = tx * ty ;
		v = pow(dishu,(float)k);

		for (i = 0;i<dy;i++)
		{
			for (j = 0;j<dx;j++)
			{
				//灰度插值图像像素
				N = w1 * (double)(unsigned char)src->imageData[(i+fy)*src->width+j+fx]+
				    w2 * (double)(unsigned char)src->imageData[(i+fy)*src->width+j+cx]+
				    w3 * (double)(unsigned char)src->imageData[(i+cy)*src->width+j+fx]+
				    w4 * (double)(unsigned char)src->imageData[(i+cy)*src->width+j+cx];

				if( N >= (double)(unsigned char)target->imageData[i*dx+j])
				{
					result[i*dx+j] = result[i*dx+j] + v * 1;
				}else{
					result[i*dx+j] = result[i*dx+j] + v * 0;
				}
			}
		}
	}
	//显示图像
	if (num_sp<=8)
	{
		//只有采样数小于8，则编码范围0-255，才能显示图像
		for (i = 0;i<dy;i++)
		{
			for (j = 0;j<dx;j++)
			{
				target->imageData[i*dx+j] = (unsigned char)result[i*dx+j];
				//printf("%d\n",(unsigned char)target->imageData[i*width+j]);
			}
		}
		cvSaveImage("/data/result.jpg",target);
	}

	//显示直方图

	for (i=0;i<cols;i++)
	{
		val_hist[i]=0.0;
	}
	for (i=0; i<dy*dx;i++)
	{
		val_hist[result[i]]+=1;
	}

	double temp_max=0.0;

	for (i=0;i<cols;i++)         //求直方图最大值，为了归一化
	{
		//printf("%f\n",val_hist[i]);
		if (temp_max<val_hist[i])
		{
			temp_max=val_hist[i];
		}
	}
	//画直方图
	CvPoint p1,p2;
	double bin_width=(double)hist->width/cols;
	double bin_unith=(double)hist->height/temp_max;

	for (i=0;i<cols;i++)
	{
		p1.x=i*bin_width;p1.y=hist->height;
		p2.x=(i+1)*bin_width;p2.y=hist->height-val_hist[i]*bin_unith;
		cvRectangle(hist,p1,p2,cvScalar(0,255),-1,8,0);
	}
	cvSaveImage("/data/hist.jpg",hist);
	return 0;
}



int main()
{
	IplImage* img = cvLoadImage( "/data/timg.jpg" );
	cvNamedWindow("Example1", 1 );
	cvShowImage("Example1", img );
	CvPoint point[img->height*img->width];
	for(int i=0;i<img->height;i++)
		for(int j=0;j<img->width;j++)
			point[i*img->width+j].x=j,point[i*img->width+j].y=i;
	if(!gray_invariant_lbp(img,img->height,img->width,100,point))
		printf("success!\n");
//	cvWaitKey(0);
	cvReleaseImage( &img );
	cvDestroyWindow("Example1");
//	Mat srcImage = imread("//data//timg.jpg");
//	imshow("[img]", srcImage);
//	waitKey(0);
//	return 0;
}