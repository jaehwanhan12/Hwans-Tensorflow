import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
#데이터 읽고 변환하는 코드
data = np.loadtxt('./data.csv',delimiter=',',
                  unpack=True,dtype='float32') # unpack=True라는 매개변수는 행렬을 전치 시키는 역할
# 털, 날개, 기타, 포유류, 조류
# x_data = 0, 1
# y_data = 2, 3, 4
x_data = np.transpose(data[0:2]) #0은 포함되지 않는다.즉, 1 2 열 #np.transpose가 전치의 역할을 하는것
y_data = np.transpose(data[2:]) #3열부터~ 마지막열
####################
####신경망 모델 구성
####################
#모델을 저장할때 쓸 변수를 만드는것!
#학습에 사용되는 것이 아니라, 학습횟수를 카운트하는 변수로 "trainable=False" 라는 옵션을 추가
global_step = tf.Variable(0,trainable=False, name='global_step')
#계층을 하나 더 : W2,L2, 편향은 없고 : b 변수 생성 X, 오로지 가중치(W)만으로 모델구성
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
W1=tf.Variable(tf.random_uniform([2,10],-1.,1.)) ##주의 점 계층간의 크기를 보고 층의 형태 만들기
L1=tf.nn.relu(tf.matmul(X,W1))
W2=tf.Variable(tf.random_uniform([10,20],-1.,1.))
L2=tf.nn.relu(tf.matmul(L1,W2))
W3 = tf.Variable(tf.random_uniform([20,3],-1.,1.))
model = tf.matmul(L2,W3)
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost,global_step=global_step) # 앞에서 정의된 global_step을 넣음으로써 최적할때마다 global_step 변수값이 1씩 증가한다 첨에 0 으로 정의했어!
###신경망 모델 학습
sess= tf.Session() # 세션을 열고 최적화하기
saver = tf.train.Saver(tf.global_variables()) # tf.global_variables 는 앞의정의된 변수를 가져오는 함수 이 함수를 통해 이 변수들을 파일에 저장하거나 이전 결과를 불러와 담는 변수들로 활용

###################위 까지, 모델구성 및 불러오기 저장 변수 형성#########################################

####체크포인트 모델생성 << 실제로 학습된 모델의 Variable 을 저장하는곳

ckpt = tf.train.get_checkpoint_state('./model') #체크포인트 파일에 대한 지정
## 체크포인트 파일이 본 폴더에 있다면 불러오고 없으면 초기화해라! (사용할거니깐)
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
    #위에서 model 이라는 학습모델 저장했고
    #아래에서 재학습한 결과를 보기위해 2번으로설정
for step in range(2):
    sess.run(train_op, feed_dict={X:x_data, Y: y_data})
    print('Step: %d,' % sess.run(global_step),
          'Cost; %.3f' % sess.run(cost, feed_dict={X:x_data, Y: y_data}))
saver.save(sess,'./model/dnn.ckpt',global_step=global_step)

### 결과 확인

prediction = tf.argmax(model,1)
target = tf.argmax(Y,1)

print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data }))

is_correct = tf.equal(prediction,target)
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))


