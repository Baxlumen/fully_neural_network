#include <bits/stdc++.h>
#define IPNNUM 784
#define HDNNUM 100
#define OPNNUM 10
using namespace std;
//node class,build net
class node
{
public:
    double value; //数值，存储结点最后的状态
    double *W = NULL;    //结点到下一层的权值

    void initNode(int num);//初始化函数，必须调用以初始化权值个数
    ~node();     //析构函数，释放掉权值占用内存
};

void node::initNode(int num)
{
    W = new double[num];
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < num; i++)//给权值赋一个随机值
    {
        W[i] = rand() % 100 / double(100)*0.1;
        if (rand() % 2)
        {
            W[i] = -W[i];
        }
    }
}
node::~node() {
    if(W!=NULL) delete []W;
}
class net
{
public:
    node inlayer[IPNNUM];
    node hidlayer[HDNNUM];
    node outlayer[OPNNUM];

    double beta = 0.2; //学习率
    double k1;
    double k2;//偏置
    double Tg[OPNNUM]; //实际值（目标值）
    double O[OPNNUM]; // 预测值

    net(); // 构造函数，用于初始化各层与偏置的权重
    double sigmoid(double z); // 激活函数
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
    //初始化输入到隐藏节点个数

    for(int i=0;i<IPNNUM;i++)
    {
        inlayer[i].initNode(HDNNUM);
    }
    //初始化隐藏到输出的个数
    for (int i = 0; i < HDNNUM; ++i) {
        hidlayer[i].initNode(OPNNUM);
    }
}
//激活函数
double net::sigmoid(double z) {
    return 1/(1+exp(-z));
}
//损失函数
double net::getloss() {
    double mloss = 0;
    for(int i=0;i<OPNNUM;i++)
    {
        mloss += pow(O[i]-Tg[i],2);
    }
    return mloss/OPNNUM;
}
//前向
void net::forward(double *input) {
    for (size_t iNNum = 0; iNNum < IPNNUM; iNNum++)//输入层节点赋值
    {
        inlayer[iNNum].value = input[iNNum];
    }
    for (size_t hNNum = 0; hNNum < HDNNUM; hNNum++)//算出隐含层结点的值
    {
        double z = 0;
        for (size_t iNNum = 0; iNNum < IPNNUM; iNNum++)
        {
            z += inlayer[iNNum].value*inlayer[iNNum].W[hNNum];
        }
        z += k1;//加上偏置项
        hidlayer[hNNum].value = sigmoid(z);
    }
    for (size_t oNNum = 0; oNNum < OPNNUM; oNNum++)//算出输出层结点的值
    {
        double z = 0;
        for (size_t hNNum = 0; hNNum < HDNNUM; hNNum++)
        {
            z += hidlayer[hNNum].value*hidlayer[hNNum].W[oNNum];
        }
        z += k2;//加上偏置项
        O[oNNum] = outlayer[oNNum].value = sigmoid(z);
    }
}
void net::backward(double *T) {
    for (size_t i = 0; i < OPNNUM; i++)
    {
        Tg[i] = T[i];
    }
    for (size_t iNNum = 0; iNNum < IPNNUM; iNNum++)//更新输入层权重
    {
        for (size_t hNNum = 0; hNNum < HDNNUM; hNNum++)
        {
            double y = hidlayer[hNNum].value;
            double loss = 0;
            for (size_t oNNum = 0; oNNum < OPNNUM; oNNum++)
            {
                loss += (O[oNNum] - Tg[oNNum])*O[oNNum] * (1 - O[oNNum])*hidlayer[hNNum].W[oNNum];
            }
            inlayer[iNNum].W[hNNum] -= beta * loss*y*(1 - y)*inlayer[iNNum].value;
        }
    }
    for (size_t hNNum = 0; hNNum < HDNNUM; hNNum++)//更新隐含层权重
    {
        for (size_t oNNum = 0; oNNum < OPNNUM; oNNum++)
        {
            hidlayer[hNNum].W[oNNum] -= beta * (O[oNNum] - Tg[oNNum])*
                                        O[oNNum] * (1 - O[oNNum])*hidlayer[hNNum].value;
        }
    }
}

void net::printresual(int trainingTimes)
{
    double loss = getloss();
    cout << "train times:" << trainingTimes << endl;
    cout << "loss:" << loss << endl;
    for (size_t oNNum = 0; oNNum < OPNNUM; oNNum++)
    {
        cout << "output" << oNNum+1<< ":" << O[oNNum] << endl;
    }
    cout<<endl;
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
        for (size_t i = 0; i < 10; i++)
        {
            if (mnet->outlayer[i].value > value)
            {
                value = mnet->outlayer[i].value;
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
for(int j=0;j<5;j++) {
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
