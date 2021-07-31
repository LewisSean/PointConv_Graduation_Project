'''
点云识别系统的客户端GUI
by liushun
'''
from tkinter import Frame, Button, StringVar, Entry, Label, IntVar, Text, END
from tkinter.ttk import Combobox
import tkinter.messagebox as messagebox
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from visual_util import my_data_load, model_load, class_names, trained_model, load_models_show, open_3d
import numpy as np
import torch
import paramiko
from demo.tk_utils import TextWithVar


class Application(Frame):

    def __init__(self, master=None):
        self.path = ''
        self.trans = paramiko.Transport(('10.201.230.200', 22))
        self.trans.connect(username='bstemp', password='09017144')
        self.ssh = paramiko.SSHClient()
        self.ssh._transport = self.trans

        Frame.__init__(self, master)
        self.pack()
        self.result = StringVar()
        self.path_var = StringVar()
        self.topkStr = StringVar(value='')
        self.topkModels = []
        self.k_val = IntVar(value=5)
        self.device = IntVar(value=0)  # 默认GPU 0
        self.embedding = StringVar(value='fcn_output')
        self.model_data = np.array([0]) # 1024 x 6
        self.input = torch.zeros(1)  # 32 x 1024 x 6
        self.classifier = model_load(trained_model)
        self.classifier = self.classifier.eval()
        self.createWidgets()

    def createWidgets(self):
        cur_row = 0
        self.topTitle = Label(self, text='Welcome, this is PCCS!')
        self.topTitle.grid(row=cur_row, column=1, sticky="ew")
        cur_row += 1
        self.alertButton = Button(self, text='get model address', command=self.selectPath_pro)
        self.alertButton.grid(row=cur_row, column=0, sticky="ew")
        self.addrInput = Entry(self, textvariable=self.path_var)
        self.addrInput.grid(row=cur_row, column=1, sticky="ew")
        self.showButton = Button(self, text='show model', command=self.show_matplotlib)
        self.showButton.grid(row=cur_row, column=2, sticky="ew")
        cur_row += 1
        self.predictButton = Button(self, text='predict', command=self.predict)
        self.predictButton.grid(row=cur_row, column=0, sticky="ew")
        self.predictText = Entry(self, textvariable=self.result)
        self.predictText.grid(row=cur_row, column=1, sticky="ew")
        cur_row += 1
        self.topkLabel = Label(self, text='choose top k value:')
        self.topkLabel.grid(row=cur_row, column=0, sticky="ew")
        self.topkBL = Combobox(self, textvariable=self.k_val)
        self.topkBL["values"] = (3, 5, 7)
        self.topkBL.current(1)
        self.topkBL.grid(row=cur_row, column=1, sticky="ew")
        cur_row += 1
        self.gpuLabel = Label(self, text='choose gpu id:')
        self.gpuLabel.grid(row=cur_row, column=0, sticky="ew")
        self.gpuBL = Combobox(self, textvariable=self.device)
        self.gpuBL["values"] = (0, 1)
        self.gpuBL.current(0)
        self.gpuBL.grid(row=cur_row, column=1, sticky="ew")
        cur_row += 1
        self.embeddingLabel = Label(self, text='choose embedding mode:')
        self.embeddingLabel.grid(row=cur_row, column=0, sticky="ew")
        self.embeddingBL = Combobox(self, textvariable=self.embedding)
        self.embeddingBL["values"] = ('fcn_output_128', 'fcn_output_256', 'PCA')
        self.embeddingBL.current(0)
        self.embeddingBL.grid(row=cur_row, column=1, sticky="ew")
        cur_row += 1
        self.topkButton = Button(self, text='search top k models', command=self.search_topk_ssh)
        self.topkButton.grid(row=cur_row, column=0, sticky="ew")
        self.topkList = TextWithVar(self, height=7, width=10, textvariable=self.topkStr)
        self.topkList.grid(row=cur_row, column=1, sticky="ew")
        self.topkShowButton = Button(self, text='show these models', command=self.show_topk)
        self.topkShowButton.grid(row=cur_row, column=2, sticky="ew")
        cur_row += 1
        self.resetButton = Button(self, text='Reset', command=self.reset)
        self.resetButton.grid(row=cur_row, column=0, sticky="ew")
        self.quitButton = Button(self, text='Quit', command=self.quit)
        self.quitButton.grid(row=cur_row, column=2, sticky="ew")

    def show_topk(self):
        path = "../data/modelnet40_normal_resampled/"+self.result.get()+'/'
        _list = []
        for fn in self.topkModels:
            _list.append(path+fn+'.txt')
        load_models_show(_list)

    def search_topk_ssh(self):
        if self.result.get() == "":
            messagebox.showinfo('Error!', 'Please PREDICT the model FIRST!')

        # 上传本地文件到服务器
        sftp = paramiko.SFTPClient.from_transport(self.trans)
        fn = self.path[self.path.rfind("/") + 1:]
        sftp.put(localpath=self.path, remotepath='LS/pointconv_pytorch/data/cache/'+fn)

        command = 'cd LS/pointconv_pytorch && python3 IR_shell.py --fn {} --topk {} --gpu {} --compress {}'.format(fn, self.k_val.get(), self.device.get(), self.embedding.get())

        print(command)
        stdin, stdout, stderr = self.ssh.exec_command(command)
        text = stdout.read().decode()
        terms = text.split('\n')

        target = text.split('\n')[-2]

        self.topkModels = target.split(' ')[1:]
        print(self.topkModels)
        if len(self.topkModels) == 0:
            self.topkStr.set("Not Found!")
        else:
            tmp = ''
            i = 1
            for fn in self.topkModels:
                tmp += "{}. {}\n".format(i, fn)
                i += 1
            self.topkStr.set(tmp)

    def show_matplotlib(self):
        open_3d(self.model_data[:, :3], window_name="show")

    def predict(self):
        points = self.input.permute(0, 2, 1)  # (32, 6, 1024)
        with torch.no_grad():
            pred = self.classifier(points[:, :3, :], points[:, 3:, :])
        pred_choice = pred.data.max(1)[1]
        self.result.set(class_names[int(pred_choice[0])])
        print("Predict: " + class_names[int(pred_choice[0])])
        messagebox.showinfo('Result', 'Predicted class: {}'.format(class_names[int(pred_choice[0])]))

    def selectPath(self):
        self.path = askopenfilename()
        while not (self.path.endswith('.txt') or self.path == ''):
            messagebox.showinfo('Warning', 'it is not a point cloud model path!')
            self.path = askopenfilename()

    def selectPath_pro(self):
        # 设置可以选择的文件类型，不属于这个类型的，无法被选中
        filetypes = [("文本文件", "*.txt")]
        self.path = filedialog.askopenfilename(title='选择单个文件', filetypes=filetypes, initialdir='../data/sample')
        self.path_var.set(self.path)
        self.model_data = my_data_load(self.path)
        self.input = torch.from_numpy(self.model_data.reshape((1, self.model_data.shape[0], self.model_data.shape[1])))

    def reset(self):
        self.path = ''
        self.model_data = np.array([0])
        self.result.set('')
        self.path_var.set('')
        self.input = torch.zeros(1)
        self.topkModels = []
        self.topkStr.set('')
        # print(self.k_val.get())

    def quit(self):
        self.trans.close()
        self.tk.quit()


app = Application()
app.master.title('点云识别综合系统')
app.mainloop()
