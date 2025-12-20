Step 1:
sudo apt update
sudo apt install -y \
  python3.11 python3.11-venv python3.11-dev \
  build-essential pkg-config cmake git \
  libgl1 libglib2.0-0 libgtk-3-0 \
  libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-render-util0 libxcb-shape0 libxcb-randr0 \
  libxcb-sync1 libxcb-xfixes0 libxcb-xkb1 libx11-xcb1 \
  libxrender1 libxi6 libxrandr2


Step 2:
python3.11 -m venv ~/rms
echo "alias rms='source ~/rms/bin/activate'" >> ~/.bashrc
source ~/.bashrc
rms  # Activate the environment


Step 3:
pip install --upgrade pip setuptools wheel
pip install "numpy<2.0" scipy pandas numba llvmlite
pip install cython matplotlib gitpython paramiko Pillow imageio
pip install pyqt5==5.15.10 pyqtgraph==0.12.3 pygobject
pip install "opencv-python-headless<4.12"
pip install ephem astropy "rawpy<0.22"
pip install "git+https://github.com/matejak/imreg_dft@master#egg=imreg_dft>2.0.0"
pip install astrometry

Step 4:
# If you have a requirements.txt
pip install -r requirements.txt

# Install your project locally without build isolation
pip install --no-build-isolation .

