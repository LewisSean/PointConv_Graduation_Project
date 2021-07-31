'''
测试 paramiko 包中的 ssh 功能
by liushun
'''

import paramiko
# 实例化一个transport对象
trans = paramiko.Transport(('10.201.230.200', 22))
# 建立连接
trans.connect(username='bstemp', password='09017144')

# 将sshclient的对象的transport指定为以上的trans
ssh = paramiko.SSHClient()
ssh._transport = trans
# 执行命令，和传统方法一样
stdin, stdout, stderr = ssh.exec_command('cd LS/pointconv_pytorch && python3 IR_shell.py')
text = stdout.read().decode()
print(text.split('\n')[-3])
print(text.split('\n')[-2])


stdin, stdout, stderr = ssh.exec_command('cd LS/pointconv_pytorch && python3 IR_shell.py')
text = stdout.read().decode()
print(text.split('\n')[-3])
print(text.split('\n')[-2])

# 关闭连接
trans.close()
