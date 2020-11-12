import subprocess
from subprocess import Popen
import os

def splitAudio(videoName, videoPath):
    if not os.path.exists(videoName.strip(".mp4") + ".m4a"):
        with open("audioLog.txt", "w") as log:
            cmd = "ffmpeg -i %s -vn -codec copy %s.m4a"%(videoName, videoName.strip(".mp4"))
            subProcess = Popen(cmd, shell=True, stdout=log, stderr=log)
            subProcess.communicate()
    print("Audio Split Done! %s"%videoName)

def mergeVideoAndAudio(videoName, videoPath):
    outputName = videoName.strip(".mp4") + "-Subtitle.mp4"
    if os.path.exists(outputName):
        os.remove(outputName)
    with open("mergeLog.txt", "w") as log:
        cmd = "ffmpeg -i %s -i %s.m4a -acodec copy -vcodec copy %s"%(videoName, videoName.strip(".mp4"), outputName)
        subProcess = Popen(cmd, shell=True, stdout=log, stderr=log)
        subProcess.communicate()
        
    print("Video Audio Merge Done! %s"%videoName)

def processOutput(videoName='demo.mp4', videoPath='/videoOutput/demo'):
    """
    :param videoPath: Relative path from os.cwd(), (That is the root dir of this proejct in default).
    :param videoName: The file we want to process.
    """
    videoPath = os.getcwd()+videoPath
    # change to the output dir, make ffmpeg easier to use
    os.chdir(videoPath)
    print(os.getcwd())
    
    splitAudio(videoName, videoPath)
    mergeVideoAndAudio(videoName, videoPath)
    print("Ouput Porcess Done! %s"%videoName)

processOutput("TWICE-What_Is_Love.mp4")