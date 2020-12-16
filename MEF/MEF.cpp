// MEF.cpp : 定义控制台应用程序的入口点。
//

// 输入：图片序列文件夹
// 输出：融合图片


#include <io.h>
#include <string>
#include <vector>


#include <opencv\highgui.h>
#include <opencv2\imgproc\imgproc.hpp>



using namespace std;
using namespace cv;

void getFiles(string path, vector<string>& files)
{
	//文件句柄
	long hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	string p;

	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之
			//如果不是,加入列表
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

int main(int argc, char* argv[])
{
	vector<string> files;

	char *img_seq_path = ".\\input\\Room";

	getFiles(img_seq_path, files);

	int frames = files.size();
	vector<Mat> imgs;
	vector<Mat> patchs;
	vector<double> y_mean;
	vector<double> fc;
	double fc_max = 0;
	int p_fs = 2;
	Mat sum_fs_s;
	Mat sum_fs;
	Mat fs_vec;
	double sum_fl_l = 0;
	double sum_fl = 0;
	double fl = 0;

	double delata = 0.5;
	int patch_size = 11;
	unsigned char patch[361] = { 0 };
	

	//read all images
	for (int k = 0; k < frames; k++)
	{
		Mat ycrcb;
		Mat img = imread(files[k]);
		cvtColor(img, ycrcb, CV_RGB2YCrCb);
		vector<Mat> vec;
		split(ycrcb, vec);
		imgs.push_back(vec[0]);
		Scalar ymean = mean(vec[0]);
		y_mean.push_back(ymean.val[0]);
		imshow("img seq", imgs[k]);
		waitKey(0);
	}

	

	int image_width = imgs[0 ].cols;
	int image_heit = imgs[0].rows;

	for (int i = patch_size / 2; i < image_heit - patch_size / 2; i++)
	{
		for (int j = patch_size / 2; j < image_width - patch_size / 2; j++)
		{
			//准备patchs[1,K]
			for (int k = 0; k < frames; k++)
			{
				Mat patch = Mat(patch_size, patch_size, CV_8U);
				patchs.push_back(patch);
				unsigned char *p1 = imgs[k].data;
				unsigned char *p2 = patch.data;
				for (int n = 0; n < patch_size; n++)
				{
					for (int m = 0; m < patch_size; m++)
					{
						p2[n*patch_size + m] = p1[(i - patch_size / 2 + n)* image_width + j - patch_size / 2 + m];
					}
				}
				imshow("patch", patch);
				waitKey(0);

				Scalar patch_mean = mean(patch);
				double lk = patch_mean.val[0];

				double n2 = norm(patch - lk, 2);
				//计算每个patch局部对比度
				if (n2 > fc_max) 
					fc_max = n2;
				fc.push_back(n2);

				//计算每个patch结构
				double np = norm(patch - lk, p_fs);
				
				Mat a = patch / n2;
				imshow("a", a);
				waitKey(0);
				sum_fs_s +=  np*(patch / n2);
				sum_fs += np;


				//计算每个patch平均亮度
				double uk = y_mean[k];
				double L_uk_lk = exp(((0 - ((uk - 0.5)*(uk - 0.5))) - (0 - ((lk - 0.5)*(lk - 0.5))))/(2*delata*delata));
				sum_fl_l += L_uk_lk*lk;
				sum_fl += L_uk_lk;

			}

			//Fc + 检查取值有没有问题
			fc_max = fc_max;

			//Fs + 检查取值有没有问题
			fs_vec = (sum_fs_s / sum_fs) / (norm((sum_fs_s / sum_fs), 2));

			//Fl + 检查取值有没有问题
			fl = sum_fl_l / sum_fl;

			imshow("out", fc_max*fs_vec + fl);
			waitKey(0);

			sum_fl_l = 0;
			sum_fl = 0;
			sum_fs.release();
			sum_fs_s.release();
			fc_max = 0;
			fc.clear();
			patchs.clear();
		}
	}




	return 0;
}

