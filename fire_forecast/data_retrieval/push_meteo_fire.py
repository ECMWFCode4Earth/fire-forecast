# push input data for Code for Earth team to ftp server at IONOS
# 230804 kaiser@satfire.org

import os
import sys
import subprocess
import pysftp
import datetime as dt
import gzip
import shutil
import glob

ftp_SatFire = {
    'host' : 'access944774485.webspace-data.io',
    'name' : 'u1301442635',
    'pw'   : 'deqsyt-qahmir-bUpse' }
    
serv = ftp_SatFire

fList = glob.glob('/scratch/cyjk/code4earth/0p5/*.nc')

print(f'push_meteo_fire will transfer {len(fList)} files: {fList}')

# sftp connection options
cnopts = pysftp.CnOpts()
cnopts.compression = True

# actual data handling
with pysftp.Connection(serv['host'], username=serv['name'], password=serv['pw'],
                       port=22, cnopts=cnopts) as sftp:

    # change to remote output directory    
    if not sftp.exists('code4earth'):
        sftp.mkdir('code4earth')
    sftp.chdir('code4earth')
    
    # cycle over files
    for fName in fList:
            sys.stdout.write(f'pushing {fName}...')
            sftp.put(fName)
                
            sys.stdout.write('done.\n')
            sys.stdout.flush()
