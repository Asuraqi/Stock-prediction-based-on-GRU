import numpy as np

def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res

def diffsigmoid(x):
    res = x*(1-x)
    return res

def outTanh(x):
    res = 1 -2 / (1 + np.exp(2*x))
    return res

def diffoutTanh(x):
    res = 1 -x**2
    return res
class Adam():
    def __init__(self, r, s, lr, beta_1=0.9, beta_2=0.999, epsilon=0.001, delta=1e-8):
        self.lr = lr
        self.beta1 = beta_1
        self.beta2 = beta_2
        self.epislon = epsilon
        self.delta = delta
        self.r = r
        self.s = s
        self.n = 0

    def __call__(self, gradient: np) -> np:
        self.n += 1

        self.r = self.beta1 * self.r + (1 - self.beta1) * gradient
        self.s = self.beta2 * self.s + (1 - self.beta2) * np.square(gradient)

        alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.n))
        alpha = alpha / (1 - np.power(self.beta1, self.n))

        return alpha * self.r / (np.sqrt(self.s) + self.epislon)

class GRULayer():
    """
    Input with Dimension 1 x x^d in the format of Either list or NumPy.
    Hidden with predefined output dimension 1 x h^d
    """
    def __init__(self, input, hidden, y_label):

        self.uNum=np.shape(input)[0]
        self.xdim=np.shape(input)[1]
        self.ydim=1 if np.shape(y_label)[0] == np.size(y_label) else np.shape(y_label)[1]
        self.hdim=np.size(hidden)

        # Reset Gate
        self.W_xr_ = np.random.rand(self.xdim,self.hdim)*2-1  # dimension x^d x h^d
        self.W_hr_ = np.random.rand(self.hdim,self.hdim)*2-1  # dimension h^d x h^d

        # Update Gate
        self.W_xz_ = np.random.rand(self.xdim,self.hdim)*2-1 # dimension x^d x h^d
        self.W_hz_ = np.random.rand(self.hdim,self.hdim)*2-1  # dimension h^d x h^d

        # H_tilda
        self.W_xs_ = np.random.rand(self.xdim,self.hdim)*2-1  # dimension x^d x h^d
        self.W_hs_ = np.random.rand(self.hdim,self.hdim)*2-1  # dimension h^d x h^d

        # Output
        self.W_hy_ = np.random.rand(self.hdim,self.ydim)*2-1 # dimension: h^d x 1

        # List Value for Retrieval
        # Input will be stored in the form of list with features
        # Each input will be in the form of 1 x x
        self.x_values = input
        self.d = y_label       #label with result will be stored

        self.rvalues = np.zeros(shape=(self.uNum+1,self.hdim))  #r
        self.zvalues = np.zeros(shape=(self.uNum+1,self.hdim))  #z
        self.svalues = np.zeros(shape=(self.uNum,self.hdim))    #s
        self.hvalues = np.zeros(shape=(self.uNum,self.hdim))    #h
        self.yvalues = np.zeros(shape=(self.uNum,))    #y


    def change_input(self,input, output):
        self.x_values = input
        self.d = output
        self.uNum = np.shape(input)[0]
        self.yvalues = np.zeros(shape=(self.uNum,))  # y

    def forward(self):
        self.rvalues[0,:]=sigmoid(np.matmul(self.x_values[0,:],self.W_xr_))
        self.svalues[0,:]=outTanh(np.matmul(self.x_values[0,:],self.W_xs_))
        self.zvalues[0,:]=sigmoid(np.matmul(self.x_values[0,:],self.W_xz_))
        self.hvalues[0,:]=np.multiply(self.zvalues[0,:],self.svalues[0,:])
        self.yvalues[0]=sigmoid(np.matmul(self.hvalues[0,:],self.W_hy_))

        for t in range(1,self.uNum):
            self.rvalues[t,:] = sigmoid(np.matmul(self.x_values[t,:],
                            self.W_xr_)+np.matmul(self.hvalues[t-1,:],self.W_hr_))
            self.svalues[t,:] = outTanh(np.matmul(self.x_values[t,:],
                            self.W_xs_)+np.matmul(np.multiply(self.rvalues[t,:],self.hvalues[t-1,:]),self.W_hs_))
            self.zvalues[t,:] = sigmoid(np.matmul(self.x_values[t,:],
                            self.W_xz_)+np.matmul(self.hvalues[t-1,:],self.W_hz_))
            self.hvalues[t,:] = np.multiply(1-self.zvalues[t,:],
                            self.hvalues[t-1,:])+np.multiply(self.zvalues[t,:],self.svalues[t,:])
            self.yvalues[t] = sigmoid(np.matmul(self.hvalues[t,:], self.W_hy_))

        return self.yvalues
    def forward2(self,h_start):
        self.rvalues[0,:]=sigmoid(np.matmul(self.x_values[0,:],self.W_xr_)+np.matmul(h_start,self.W_hr_))
        self.svalues[0,:]=outTanh(np.matmul(self.x_values[0,:],self.W_xs_)+np.matmul(np.multiply(self.rvalues[0,:],h_start),self.W_hs_))
        self.zvalues[0,:]=sigmoid(np.matmul(self.x_values[0,:],self.W_xz_)+np.matmul(h_start,self.W_hz_))
        # self.hvalues[0,:]=np.multiply(self.zvalues[0,:],self.svalues[0,:])
        self.hvalues[0, :]=h_start
        self.yvalues[0]=sigmoid(np.matmul(self.hvalues[0,:],self.W_hy_))

        for t in range(1,self.uNum):
            self.rvalues[t,:] = sigmoid(np.matmul(self.x_values[t,:],
                            self.W_xr_)+np.matmul(self.hvalues[t-1,:],self.W_hr_))
            self.svalues[t,:] = outTanh(np.matmul(self.x_values[t,:],
                            self.W_xs_)+np.matmul(np.multiply(self.rvalues[t,:],self.hvalues[t-1,:]),self.W_hs_))
            self.zvalues[t,:] = sigmoid(np.matmul(self.x_values[t,:],
                            self.W_xz_)+np.matmul(self.hvalues[t-1,:],self.W_hz_))
            self.hvalues[t,:] = np.multiply(1-self.zvalues[t,:],
                            self.hvalues[t-1,:])+np.multiply(self.zvalues[t,:],self.svalues[t,:])
            self.yvalues[t] = sigmoid(np.matmul(self.hvalues[t,:], self.W_hy_))

        return self.hvalues
    def reset(self):
        self.rvalues = np.zeros(shape=(self.uNum+1,self.hdim))  # r
        self.zvalues = np.zeros(shape=(self.uNum+1,self.hdim))  # z
        self.svalues = np.zeros(shape=(self.uNum,self.hdim))  # s
        self.hvalues = np.zeros(shape=(self.uNum,self.hdim))  # h
        self.yvalues = np.zeros(shape=(self.uNum,))  # y

    def backprop(self,w,y_G,learning_rate=1e-4,optimizer='Adam'):
        delta_r_next = np.zeros(shape=(1, self.hdim))
        delta_z_next = np.zeros(shape=(1, self.hdim))
        delta_h_next = np.zeros(shape=(1, self.hdim))
        delta_next = np.zeros(shape=(1, self.hdim))

        dWhy = np.zeros(shape=(self.hdim, self.ydim))
        dWxr = np.zeros(shape=(self.xdim, self.hdim))
        dWhr = np.zeros(shape=(self.hdim, self.hdim))
        dWxs = np.zeros(shape=(self.xdim, self.hdim))
        dWhs = np.zeros(shape=(self.hdim, self.hdim))
        dWxz = np.zeros(shape=(self.xdim, self.hdim))
        dWhz = np.zeros(shape=(self.hdim, self.hdim))
        s_hy = np.zeros(shape=np.shape(dWhy))
        s_xr = np.zeros(shape=np.shape(dWxr))
        s_hr = np.zeros(shape=np.shape(dWhr))
        s_xs = np.zeros(shape=np.shape(dWxs))
        s_hs = np.zeros(shape=np.shape(dWhs))
        s_xz = np.zeros(shape=np.shape(dWxz))
        s_hz = np.zeros(shape=np.shape(dWhz))

        r_hy = np.zeros(shape=np.shape(dWhy))
        r_xr = np.zeros(shape=np.shape(dWxr))
        r_hr = np.zeros(shape=np.shape(dWhr))
        r_xs = np.zeros(shape=np.shape(dWxs))
        r_hs = np.zeros(shape=np.shape(dWhs))
        r_xz = np.zeros(shape=np.shape(dWxz))
        r_hz = np.zeros(shape=np.shape(dWhz))

        if optimizer == 'Adam':
            adam_hy = Adam(r_hy, s_hy, learning_rate)
            adam_xr = Adam(r_xr, s_xr, learning_rate)
            adam_hr = Adam(r_hr, s_hr, learning_rate)
            adam_xs = Adam(r_xs, s_xs, learning_rate)
            adam_hs = Adam(r_hs, s_hs, learning_rate)
            adam_xz = Adam(r_xz, s_xz, learning_rate)
            adam_hz = Adam(r_hz, s_hz, learning_rate)
        else:
            None
        t = self.uNum-1
        while t > 0:
            delta_y = w*(y_G[t] - self.d[t])*diffsigmoid(self.yvalues[t])
            delta_h = delta_y*np.transpose(self.W_hy_)+np.matmul(delta_z_next,np.transpose(self.W_hz_))+np.multiply(
                    np.matmul(delta_next,np.transpose(self.W_hs_)), self.rvalues[t+1,:])+np.matmul(delta_r_next,
                    np.transpose(self.W_hr_))+np.multiply(delta_h_next,1-self.zvalues[t+1,:])
            t1 = np.multiply(delta_h, self.svalues[t,:]-self.svalues[t-1,:])
            delta_z = np.multiply(t1, diffsigmoid(self.zvalues[t,:]))
            delta = np.multiply(np.multiply(delta_h,self.zvalues[t,:]),diffoutTanh(self.svalues[t,:]))
            t2 = np.multiply(self.hvalues[t-1,:], np.matmul(np.multiply(np.multiply(delta_h,
                    self.zvalues[t,:]), diffoutTanh(self.svalues[t,:])),np.transpose(self.W_hs_)))
            delta_r = np.multiply(t2, diffsigmoid(self.rvalues[t,:]))

            dWhy = dWhy + (self.hvalues[t]*delta_y).reshape(np.size(dWhy),1)
            dWxz = dWxz + self.x_values[t].reshape(np.size(self.x_values[t]),1)*delta_z
            dWhz = dWhz + np.matmul(self.hvalues[t-1], np.transpose(delta_z))
            dWxs = dWxs + np.matmul(self.x_values[t].reshape(np.size(self.x_values[t]),1), delta)
            dWhs = dWhs + np.matmul(np.multiply(self.rvalues[t],
                        self.hvalues[t-1]).reshape(np.size(self.rvalues[t]),1), delta)
            dWxr = dWxr + np.matmul(np.transpose(self.x_values[t]).reshape(np.size(self.x_values[t]),1), delta_r)
            dWhr = dWhr + np.matmul(np.transpose(self.hvalues[t-1]).reshape(np.size(self.hvalues[t]),1), delta_r)

            delta_r_next = delta_r
            delta_z_next = delta_z
            delta_h_next = delta_h
            delta_next = delta
            t = t - 1



        delta_y = w*(y_G[0]-self.d[0])*diffsigmoid(self.yvalues[0])
        delta_h = delta_y*np.transpose(self.W_hy_)+np.matmul(delta_z_next,
                    np.transpose(self.W_hz_))+np.multiply(np.matmul(delta_next,np.transpose(self.W_hs_)),
                    self.rvalues[1,:])+np.matmul(delta_r_next,np.transpose(self.W_hr_))+np.multiply(delta_h_next,
                    (1-self.zvalues[1,:]))
        delta_z = np.multiply(np.multiply(delta_h,self.svalues[0,:]),diffsigmoid(self.zvalues[0,:]))
        delta = np.multiply(np.multiply(delta_h,self.zvalues[0,:]),diffoutTanh(self.svalues[t,:]))

        dWhy = dWhy + (self.hvalues[0,:].reshape(np.size(self.hvalues[0,:]),1)*delta_y).reshape(np.size(dWhy),1)
        dWxz = dWxz + np.matmul(self.x_values[0,:].reshape(np.size(self.x_values[0,:]),1), delta_z)
        dWxs = dWxs + np.matmul(np.transpose(self.x_values[0,:]).reshape(np.size(self.x_values[0,:]),1), delta)


        if optimizer == 'Adam':
            self.W_hy_ -= adam_hy(dWhy)
            self.W_xs_ -= adam_xs(dWxs)
            self.W_hs_ -= adam_hs(dWhs)
            self.W_hz_ -= adam_hz(dWhz)
            self.W_xz_ -= adam_xz(dWxz)
            self.W_hz_ -= adam_hz(dWhz)
            self.W_xr_ -= adam_xr(dWxr)
            self.W_hr_ -= adam_hr(dWhr)
        else:
            self.W_hy_ -= learning_rate * dWhy
            self.W_xs_ -= learning_rate * dWxs
            self.W_hs_ -= learning_rate * dWhs
            self.W_hz_ -= learning_rate * dWhz
            self.W_xz_ -= learning_rate * dWxz
            self.W_hz_ -= learning_rate * dWhz
            self.W_xr_ -= learning_rate * dWxr
            self.W_hr_ -= learning_rate * dWhr


