#encoding:utf-8

from .importer import *

def make_datapath_list(target_path):
	#データセットを読み込む
	path_list = []#データセットのファイルパスのリストを作り、戻り値とする
	for path in glob.glob(target_path,recursive=True):
		path_list.append(path)
		##読み込むパスを全部表示　必要ならコメントアウトを外す
		#print(path)
	#読み込むことになる音声データの数を表示
	print("sounds : " + str(len(path_list)))
	return path_list

class GAN_Sound_Dataset(data.Dataset):
	#音声のデータセットクラス
	def __init__(self,file_list,device,batch_size,sound_length=65536,sampling_rate=16000,dat_threshold=1100):
		#file_list     : 読み込む音声のパスのリスト
		#device        : gpuで処理するかどうかを決める
		#batch_size    : バッチサイズ
		#sound_length  : 学習に用いる音の長さ
		#sampling_rate : 音声を読み込む際のサンプリングレート
		#dat_threshold : データセットのファイルの総数がdat_threshold以下ならファイルの内容を保持する
		self.file_list = file_list
		self.device = device
		self.batch_size = batch_size
		self.sound_length = sound_length
		self.sampling_rate = sampling_rate
		self.dat_threshold = dat_threshold
		#データセットのファイルの総数がdat_threshold以下ならファイルの内容を保持する
		if(len(self.file_list)<=dat_threshold):
			self.file_contents = []
			for file_path in self.file_list:
				#soundはnumpy.ndarrayで、時系列の音のデータが格納される
				sound,_ = librosa.load(file_path,sr=self.sampling_rate)
				self.file_contents.append(sound)

	#バッチサイズ, ファイルの総数のうち大きい方を返す
	def __len__(self):
		return max(self.batch_size, len(self.file_list))
	#前処理済み音声の、Tensor形式のデータを取得
	def __getitem__(self,index):
		if(len(self.file_list)<=self.dat_threshold):
			sound = self.file_contents[index%len(self.file_list)]
		else:
			#パスのリストから1つ取り出す
			sound_path = self.file_list[index%len(self.file_list)]
			#soundはnumpy.ndarrayで、時系列の音のデータが格納される
			import librosa
			sound,_ = librosa.load(sound_path,sr=self.sampling_rate)
		#Tensor形式に変換
		sound = (torch.from_numpy(sound.astype(np.float32)).clone()).to(self.device)
		#時系列の音のデータの内、大きさが1より大きい要素が存在するなら、それが1になるよう正規化
		max_amplitude = torch.max(torch.abs(sound))
		if max_amplitude > 1:
			sound /= max_amplitude
		#読み込んだ音の長さをloaded_sound_lengthとする
		loaded_sound_length = sound.shape[0]
		#読み込んだ音の長さがsound_length以下なら、
		#音の前後を0埋めして長さをself.sound_lengthに揃える
		if loaded_sound_length < self.sound_length:
			padding_length = self.sound_length - loaded_sound_length
			left_zeros = torch.zeros(padding_length//2).to(self.device)
			right_zeros = torch.zeros(padding_length - padding_length//2).to(self.device)
			sound = torch.cat([left_zeros,sound,right_zeros],dim=0).to(self.device)
			loaded_sound_length = self.sound_length
		#学習に用いる音の長さ分、読み込んだ音声からランダムな箇所を選んで切り出す
		if loaded_sound_length > self.sound_length:
			#切り出す開始箇所をランダムに選び出す
			start_index = torch.randint(0,(loaded_sound_length-self.sound_length)//2,(1,1))[0][0].item()
			end_index = start_index + self.sound_length
			sound = sound[start_index:end_index]
		#この時点ではsound.shapeはtorch.Size([3, self.sound_length])となっているが、
		#これをtorch.Size([3, 1, self.sound_length])に変換する
		sound = sound.unsqueeze(0)
		return sound

#生成された音声の出力用関数
def save_sounds(path,sounds,sampling_rate):
	now_time = time.time()
	for i,sound in enumerate(sounds):
		sound = sound.squeeze(0)
		sound = sound.to('cpu').detach().numpy().copy()
		hash_string = hashlib.md5(str(now_time).encode()).hexdigest()
		file_path = os.path.join(path,f"generated_sound_{i}_{hash_string}.wav")
		print(file_path)
		sf.write(file_path,sound,sampling_rate,format="WAV")

#動作確認
# train_wav_list = make_datapath_list('../dataset/**/*.wav')

# batch_size = 3
# dataset = GAN_Sound_Dataset(file_list=train_wav_list,device="cpu",batch_size=batch_size)

# dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

# batch_iterator = iter(dataloader)
# sounds = next(batch_iterator)
# save_sounds(sounds,16000)




