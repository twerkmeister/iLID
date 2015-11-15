import features as speechfeatures

def generate(samplerate, signal):
  feat,energy = speechfeatures.fbank(signal,samplerate,winlen=0.02,winstep=0.01,
            nfilt=40,nfft=512,lowfreq=100,highfreq=5000,preemph=0.97)

  logmel = np.log10(feat) 

  #setting figure size and working around inches
  xPix = 600
  yPix = 40
  xSize = 10 #inches
  ySize = xSize/float(xPix)*yPix
  fig = plt.figure(figsize=(xSize, ySize), dpi=xSize/xPix)
  
  #disabling axes
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)

  colormap="gray"
  plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
  # should return the image or save it instead
  plt.show()