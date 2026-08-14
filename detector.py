import json
import numpy as np
class detector():
	def __init__(self,json_fn = None,detector_name = None,distance = None,wavelength= None,shape = None,centeri=None,dqi = None,centerj=None,dqj = None,mask = None,qi_label = 'i',qj_label='j'):
		if json_fn is not None:
			print('Importing detector config from jsons...')
			self.init_from_json(json_fn)
			self.qi_label = 'i'
			self.qj_label = 'j'
		else:
			# print('Setting detector config from manual entry')
			self.detector_name = detector_name #i.e 'Pilatus_1M'
			self.distance = distance #meters
			self.wavelength = wavelength #meters
			self.shape = shape
			self.centeri = centeri #pixel shape
			self.centerj = centerj #pixel shape
			self.dqi = dqi #1/A
			self.dqj = dqj #1/A
			self.qi_label = qi_label
			self.qj_label = qj_label
		self.mask = mask
		self.make_Q()

	def __str__(self):
		print("Detector: {}".format(self.detector_name))
		print("distance: {} meters".format(self.distance))
		print("wavelength: {} meters".format(self.wavelength))
		print("image shape: {} ".format(self.shape))
		print("centeri: {}".format(self.centeri))
		print("centerj: {}".format(self.centerj))
		return super().__str__()

	def init_from_json(self,json_fn):
		#Takes in a pyFAI json and imports
		with open(json_fn, "r") as file:
			self.js_data = json.load(file)
		self.detector_name = self.js_data['poni']['detector']
		self.distance = self.js_data['poni']['dist']
		self.wavelength = self.js_data['poni']['wavelength'] *1e10 #pyFAI outputs in meters. convert to angstrom
		self.shape = self.js_data['shape']
		self.dqi = self.js_data['poni']['detector_config']['pixel1']
		self.centeri = self.js_data['poni']['poni1'] / self.dqi
		self.dqj = self.js_data['poni']['detector_config']['pixel2']
		self.centerj = self.js_data['poni']['poni2'] / self.dqj

		self.__repr__()


	def make_Q(self):
		self.qi = (np.arange(0,self.shape[0]) - self.centeri)*self.dqi 
		self.qi = 2*np.pi/self.wavelength*self.qi/self.distance 
		self.qj = (np.arange(0,self.shape[1]) - self.centerj)*self.dqj 
		self.qj = 2*np.pi/self.wavelength*self.qj/self.distance 

		# qqi,qqj = np.meshgrid(self.qi,self.qj,indexing = 'ij')
		# self.Q = np.sqrt(qqi**2+ qqj**2)
		# self.Q /= 1e10 #m to angstrom

if __name__ == '__main__':
	det = detector('saxs_4m_pyfai.json')
