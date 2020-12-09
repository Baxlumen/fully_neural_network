#include "head.h"


//node class,build net
class node
{
public:
    //double *value = new double[IPNNUM]; //数值，存储结点最后的状态
    double *value = NULL;
    double *W = NULL;    //结点到下一层的权值
    void initNode(int num1,int num2);//初始化函数，必须调用以初始化权值个数
    
};

void node::initNode(int num1,int num2)
{
    double *host_W = new double[num1*num2];
    cudaMalloc((void**)&value,sizeof(double)*num1);
    cudaMalloc((void**)&W,sizeof(double)*num1*num2);
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < num1*num2; i++)//给权值赋一个随机值
    {
        host_W[i] = rand() % 100 / double(100)*0.1;
        if (rand() % 2)
        {
            host_W[i] = -host_W[i];
        }
    }
    cudaMemcpy(W,host_W,sizeof(double)*num1*num2,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

class net
{
public:
    node inlayer;
    node hidlayer;
    node outlayer;
    cublasHandle_t handle;
    double beta = 0.001; //学习率
    double k1;
    double k2;//偏置
    double *Tg=NULL; //实际值（目标值）
    double *O=NULL; // 预测值
    double rate;
    net(); // 构造函数，用于初始化各层与偏置的权重
    double getloss(); // 损失函数
    void forward(double *input);//前向
    void backward(double *T);//反向
    void printresual(int trainingTime);//打印信息

};
net::net() {
    //初始化输入与隐藏偏置权重
    srand((unsigned)time(NULL));
    k1= (rand() % 100) / (double)100;
    k2 = (rand() % 100) / (double)100;
    rate = 0.2; //学习率
    O = new double[OPNNUM];
    //初始化输入到隐藏节点个数
    //init matrix W1 IPNNUM*HDNNUM
    inlayer.initNode(IPNNUM,HDNNUM);
    hidlayer.initNode(HDNNUM,OPNNUM);
    outlayer.initNode(OPNNUM,OPNNUM);
    cudaMalloc((void**)&Tg,sizeof(double)*OPNNUM);
    cublasCreate(&handle);

}

//损失函数
double net::getloss() {
    double mloss = 0;
    mloss = loss_gpu(Tg,outlayer.value,OPNNUM,handle);  
    return mloss;
}
//前向
void net::forward(double *input) {

    cudaMemcpy(inlayer.value,input,sizeof(double)*IPNNUM,cudaMemcpyHostToDevice);
    //hidlayer.value = inlayer.W * inlayer.value;
    // 激活函数
    //outlayer,value = hidlayer.W * hidlayer,value;
    for_cuda(inlayer.value,inlayer.W,hidlayer.value,hidlayer.W,outlayer.value,IPNNUM,HDNNUM,OPNNUM,handle);
}
void net::backward(double *T) {
    cudaMemcpy(Tg,T,sizeof(double)*OPNNUM,cudaMemcpyHostToDevice);
    back_cuda(Tg,outlayer.value,hidlayer.value,hidlayer.W,inlayer.value,inlayer.W,IPNNUM,HDNNUM,OPNNUM,rate,handle);

}

void net::printresual(int trainingTimes)
{
    double loss = getloss();
    cout << "train times:" << trainingTimes << endl;
    cout << "loss:" << loss << endl;
    
}

class ImgData//单张图像
{
public:
    unsigned char tag;
    double data[IPNNUM];
    double label[OPNNUM];
};

class getImg {
public:
    ImgData* mImgData;
    void imgTrainDataRead(const char *datapath, const char *labelpath);
    ~getImg();
};

void getImg::imgTrainDataRead(const char *datapath, const char *labelpath)
{
    /***********读取图片数据***********/
    unsigned char readbuf[4];//信息数据读取空间
    FILE *f;
    f = fopen(datapath, "rb");
    fread(readbuf,1, 4, f);//读取魔数，即文件标志位
    fread(readbuf,1, 4, f);//读取数据集图像个数
    int sumOfImg = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像个数
    fread(readbuf, 1, 4, f);//读取数据集图像行数
    int imgheight = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像行数
    fread(readbuf, 1, 4, f);//读取数据集图像列数
    int imgwidth = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像列数
    mImgData = new ImgData[sumOfImg];
    unsigned char *data = new unsigned char[IPNNUM];
    for (int i = 0; i < sumOfImg; i++)
    {
        fread(data, 1, IPNNUM, f);//读取数据集图像列数
        for (size_t px = 0; px < IPNNUM; px++)//图像数据归一化
        {
            mImgData[i].data[px] = data[px]/(double)255*0.99+0.01;
        }
    }
    delete[]data;
    fclose(f);
    /***********读取标签数据***********/
    f=fopen(labelpath, "rb");
    fread(readbuf, 1, 4, f);//读取魔数，即文件标志位
    fread(readbuf, 1, 4, f);//读取数据集图像个数
    sumOfImg = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像个数
    for (int i = 0; i < sumOfImg; i++)
    {
        fread(&mImgData[i].tag, 1, 1, f);//读取数据集图像列数
        for (size_t j = 0; j < 10; j++)
        {
            mImgData[i].label[j] = 0.01;
        }
        mImgData[i].label[mImgData[i].tag] = 0.99;
    }
    fclose(f);
}
getImg::~getImg()
{
    delete[]mImgData;
}
void AccuracyRate(int time, net *mnet, getImg *mImg)//精确率评估
{
    double tagright = 0;//正确个数统计
    for (size_t count = 0; count < 10000; count++)
    {
        mnet->forward(mImg->mImgData[count].data);//前向传播
        double value = -100;
        int gettag = -100;
        cudaMemcpy(mnet->O,mnet->outlayer.value,sizeof(double)*OPNNUM,cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < 10; i++)
        {
            if (mnet->O[i] > value)
            {
                value = mnet->O[i];
                gettag = i;
            }
        }
        if (mImg->mImgData[count].tag == gettag)
        {
            tagright++;
        }
    }
    //mnet.printresual(0);//信息打印
    cout << "num." << time + 1 << ":  ";
    cout << "zheng que lv:" << tagright / 10000 << endl;
}
int main() {

    getImg mGetTrainImg;
    //mGetTrainImg.imgTrainDataRead("D:\\IIP\\TensorNet\\decomposition\\Test_fully_1207\\mnist\\train-images.idx3-ubyte", "D:\\IIP\\TensorNet\\decomposition\\Test_fully_1207\\mnist\\train-labels.idx1-ubyte");
    mGetTrainImg.imgTrainDataRead("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    getImg mGetTestImg;
    //mGetTestImg.imgTrainDataRead("D:\\IIP\\TensorNet\\decomposition\\Test_fully_1207\\mnist\\t10k-images.idx3-ubyte", "D:\\IIP\\TensorNet\\decomposition\\Test_fully_1207\\mnist\\t10k-labels.idx1-ubyte");
    mGetTestImg.imgTrainDataRead("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
    net mnet;
    //warmup();
    for(int j=0;j<3;j++)
    {
    for (int i = 0; i < 60000; ++i) {
        mnet.forward(mGetTrainImg.mImgData[i].data);
        mnet.backward(mGetTrainImg.mImgData[i].label);
        if (i % 10000 == 0)
            mnet.printresual(i);//信息打印
    }
    AccuracyRate(j, &mnet, &mGetTestImg);
}
    return 0;
}
