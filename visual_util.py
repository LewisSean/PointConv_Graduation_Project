'''
1.最远点采样
2.open3d和matplotlib对点云的可视化
3.加载pointconv模型、加载点云实例模型、输入并预测
'''
from tqdm import tqdm
from model.pointconv import PointConvDensityClsSsg as PointConvClsSsg
import time
import numpy as np
import torch.utils.data
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import matplotlib.pyplot as plt
import open3d


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.shape
    centroids = np.zeros((B, npoint), dtype=np.long)
    distance = np.ones((B, N), dtype=np.float) * 1e5
    farthest = np.zeros(B, dtype=np.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[:, farthest, :].reshape((B, 1, 3))
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]   # 更新所有点到已采样点集合的最短距离
        farthest = np.argmax(distance, -1)  # 选取其中距离最远点作为下一个采样点
    return centroids


def plot_3d(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0],  # x
               points[:, 2],  # y
               points[:, 1],  # z
               c=points[:, 2],  # height data for color
               cmap='Spectral',
               marker="o")
    ax.axis('auto')
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()


def open_3d(source_data, window_name='由红到蓝，模型相似度降低'):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(source_data)
    open3d.visualization.draw_geometries([point_cloud], width=960, height=540, left=480, top=270, window_name=window_name)


def model_load(model="./pointconv_modelnet40-0.916126-0049.pth"):
    '''MODEL LOADING'''
    num_class = 40
    classifier = PointConvClsSsg(num_class)
    checkpoint = torch.load(model, map_location=torch.device("cpu"))
    classifier.load_state_dict(checkpoint['model_state_dict'])
    return classifier


# 读取点云txt文件并最远点采样1024个点
def my_data_load(fn="./data/sample/airplane_0002.txt", num=2048):
    with open(fn, 'r') as f:
        raw = f.readlines()
        for i in range(len(raw)):
            raw[i] = raw[i][:-1].split(",")
            for j in range(6):
                raw[i][j] = float(raw[i][j])
        data = np.array(raw).astype(np.float32)
        data = data.reshape((1, data.shape[0], data.shape[1]))
        sampled = farthest_point_sample(data[:, :, :3], num)
        return data[0, sampled, :].reshape(num, 6)

def my_tqdm_eval(model):
    '''MODEL EVA'''
    DATA_PATH = './data/modelnet40_normal_resampled/'
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='test', normal_channel=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=32, shuffle=False,
                                                 num_workers=0)
    classifier = model_load(model)
    classifier = classifier.eval()
    mean_correct = []
    i = 0
    for batch_id, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
        i = i + 1
        print("\n---iter {}".format(i))
        start_time = time.time()
        pointcloud, target = data
        target = target[:, 0]  # shape from (32,1) to (32,)

        points = pointcloud.permute(0, 2, 1)  # (32, 6, 1024)
        points, target = points, target
        with torch.no_grad():
            pred = classifier(points[:, :3, :], points[:, 3:, :])

        demo = np.transpose(points[31, :3, :].numpy(), (1, 0))  # (3, 1024) -> (1024, 3)
        plot_3d(demo)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()

        print("\n")
        for i in range(target.shape[0]):
            print("GT: " + class_names[int(target[i])] + "  ||  Predict: " + class_names[int(pred_choice[i])])

        mean_correct.append(correct.item() / float(points.size()[0]))
        print("cost: {} sec".format(time.time() - start_time))

    accuracy = np.mean(mean_correct)
    print('Total Accuracy: %f' % accuracy)


trained_model = "C:/Users/Sean/Desktop/pointconv_pytorch-master/pointconv_modelnet40-0.916126-0049.pth"
class_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
                   'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
                   'laptop',
                   'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
                   'sofa', 'stairs',
                   'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']


def load_models_show(list, num=400):
    data = []
    offset = 0
    for fn in list:
        model = my_data_load(fn, num)[:, :3]
        model[:, 2] += offset  # Z 轴
        offset += 2.2
        data.append(model)

    data = np.array(data).astype(np.float32).reshape((len(data) * num, 3))
    data[:, 2] = -data[:, 2]
    open_3d(data)


if __name__ == '__main__':

    # data = my_data_load(num=512)
    # plot_3d(data)
    # my_tqdm_eval(trained_model)

    fnlist = ['airplane_0699', 'airplane_0650', 'airplane_0682', 'airplane_0705', 'airplane_0695']  # 无翼尖为top 1
    # fnlist = ['airplane_0699'] # 无旋翼
    prefix = './data/modelnet40_normal_resampled/airplane/'
    for i in range(len(fnlist)):
        fnlist[i] = prefix + fnlist[i] + '.txt'

    load_models_show(fnlist)

    """
    with open("data/modelnet40_normal_resampled/modelnet40_test.txt","r") as f:
        fns = f.readlines()
        index32 = 0
        batch = 1
        fn_dict = {}
        class_batch_dict = {}
        for key in class_names:
            class_batch_dict[key] = set()
        for fn in fns:
            fn_dict[fn[:-1]] = "{} {}".format(batch, index32)
            class_batch_dict[fn[:fn.rfind("_")]].add(batch)
            index32 += 1
            if index32 == 32:
                batch += 1
                index32 = 0

    for k in fn_dict.keys():
        print(k + " " + fn_dict[k])

    for k in class_batch_dict.keys():
        print(k + " " + str(class_batch_dict[k]))
    """