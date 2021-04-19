#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.1.2),
    on Sat Apr 17 15:49:58 2021
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2021.1.2'
expName = 'untitled'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/yeganeh/Documents/LearningCenter/ELTE/Semester 6/Psychopy/test_2Bdeleted/test_otka_builder.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=(1024, 768), fullscr=True, screen=0, 
    winType='pyglet', allowGUI=True, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "trial_desc"
trial_descClock = core.Clock()
from numpy.random import choice
from datetime import datetime
import datetime as dt
timestamp = dt.datetime.now().isoformat()
title = visual.TextStim(win=win, name='title',
    text='',
    font='Arial',
    pos=(0, .4), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
procedures = visual.TextStim(win=win, name='procedures',
    text='',
    font='Arial',
    pos=(0, .1), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
variants = visual.TextStim(win=win, name='variants',
    text='',
    font='Arial',
    pos=(0, -.2), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-3.0);
key_res = visual.TextStim(win=win, name='key_res',
    text='(Continue by pressing any key)',
    font='Arial',
    pos=(0, -.4), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-4.0);
key_resp = keyboard.Keyboard()

# Initialize components for Routine "trial_pre_quest"
trial_pre_questClock = core.Clock()
rating = visual.RatingScale(win=win, name='rating', marker='triangle', size=1.5, pos=[0.0, 0.0], low=0, high=10, labels=['Egyáltalán nem kerülnék hipnózisba', ' Nagyon mély hipnózisba kerülnék'], scale='Ha ezt a módszert használná, milyen mély hipnózisba kerülne?')

# Initialize components for Routine "trial"
trialClock = core.Clock()

# Initialize components for Routine "trial_quest"
trial_questClock = core.Clock()
win.allowStencil = True
form = visual.Form(win=win, name='form',
    items='trial_quest.csv',
    textHeight=0.03,
    randomize=False,
    size=(1, 0.7),
    pos=(0, 0),
    style=['dark'],
    itemPadding=0.05,)
button = visual.ButtonStim(win, 
   text='Continue', font='Arvo',
   pos=(0.0, -0.45),
   letterHeight=0.05,
   size=None, borderWidth=0.0,
   fillColor='darkgrey', borderColor=None,
   color='white', colorSpace='rgb',
   opacity=None,
   bold=True, italic=False,
   padding=None,
   anchor='center',
   name='button')
button.buttonClock = core.Clock()

# Initialize components for Routine "text_posthyp"
text_posthypClock = core.Clock()
text = visual.TextStim(win=win, name='text',
    text='Kutatás-végi alapszint mérés\nMost hogy a kutatás végéhez értünk, szeretnénk az agyi aktivitásáról újra egy mérést készíteni a normál éber állapotában. A következő percekben arra kérjük majd, hogy csukott szemmel pihenjen a székben ülve, hogy az agyi aktivitását megmérhessük.\nNyomja meg a gombot amikor készen áll erre.',
    font='Arial',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
key_resp_4 = keyboard.Keyboard()

# Initialize components for Routine "posthyp_baseline"
posthyp_baselineClock = core.Clock()
posthyp = visual.MovieStim3(
    win=win, name='posthyp',
    noAudio = False,
    filename='hun-posthyp.mp4',
    ori=0, pos=(0, 0), opacity=1,
    loop=False,
    depth=0.0,
    )

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# set up handler to look after randomisation of conditions etc
loop1 = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('groupNum.xlsx', selection=choice(4, size=1)),
    seed=None, name='loop1')
thisExp.addLoop(loop1)  # add the loop to the experiment
thisLoop1 = loop1.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisLoop1.rgb)
if thisLoop1 != None:
    for paramName in thisLoop1:
        exec('{} = thisLoop1[paramName]'.format(paramName))

for thisLoop1 in loop1:
    currentLoop = loop1
    # abbreviate parameter names if possible (e.g. rgb = thisLoop1.rgb)
    if thisLoop1 != None:
        for paramName in thisLoop1:
            exec('{} = thisLoop1[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    loop2 = data.TrialHandler(nReps=1, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions(group),
        seed=None, name='loop2')
    thisExp.addLoop(loop2)  # add the loop to the experiment
    thisLoop2 = loop2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop2.rgb)
    if thisLoop2 != None:
        for paramName in thisLoop2:
            exec('{} = thisLoop2[paramName]'.format(paramName))
    
    for thisLoop2 in loop2:
        currentLoop = loop2
        # abbreviate parameter names if possible (e.g. rgb = thisLoop2.rgb)
        if thisLoop2 != None:
            for paramName in thisLoop2:
                exec('{} = thisLoop2[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "trial_desc"-------
        continueRoutine = True
        # update component parameters for each repeat
        title.setText(procedure_name)
        procedures.setText(procedure)
        variants.setText(variant)
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # keep track of which components have finished
        trial_descComponents = [title, procedures, variants, key_res, key_resp]
        for thisComponent in trial_descComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        trial_descClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "trial_desc"-------
        while continueRoutine:
            # get current time
            t = trial_descClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=trial_descClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *title* updates
            if title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                title.frameNStart = frameN  # exact frame index
                title.tStart = t  # local t and not account for scr refresh
                title.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(title, 'tStartRefresh')  # time at next scr refresh
                title.setAutoDraw(True)
            
            # *procedures* updates
            if procedures.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                procedures.frameNStart = frameN  # exact frame index
                procedures.tStart = t  # local t and not account for scr refresh
                procedures.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(procedures, 'tStartRefresh')  # time at next scr refresh
                procedures.setAutoDraw(True)
            
            # *variants* updates
            if variants.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                variants.frameNStart = frameN  # exact frame index
                variants.tStart = t  # local t and not account for scr refresh
                variants.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(variants, 'tStartRefresh')  # time at next scr refresh
                variants.setAutoDraw(True)
            
            # *key_res* updates
            if key_res.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_res.frameNStart = frameN  # exact frame index
                key_res.tStart = t  # local t and not account for scr refresh
                key_res.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_res, 'tStartRefresh')  # time at next scr refresh
                key_res.setAutoDraw(True)
            
            # *key_resp* updates
            waitOnFlip = False
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=None, waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_descComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "trial_desc"-------
        for thisComponent in trial_descComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        loop2.addData('title.started', title.tStartRefresh)
        loop2.addData('title.stopped', title.tStopRefresh)
        loop2.addData('procedures.started', procedures.tStartRefresh)
        loop2.addData('procedures.stopped', procedures.tStopRefresh)
        loop2.addData('variants.started', variants.tStartRefresh)
        loop2.addData('variants.stopped', variants.tStopRefresh)
        loop2.addData('key_res.started', key_res.tStartRefresh)
        loop2.addData('key_res.stopped', key_res.tStopRefresh)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        loop2.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            loop2.addData('key_resp.rt', key_resp.rt)
        loop2.addData('key_resp.started', key_resp.tStartRefresh)
        loop2.addData('key_resp.stopped', key_resp.tStopRefresh)
        # the Routine "trial_desc" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # ------Prepare to start Routine "trial_pre_quest"-------
        continueRoutine = True
        # update component parameters for each repeat
        rating.reset()
        # keep track of which components have finished
        trial_pre_questComponents = [rating]
        for thisComponent in trial_pre_questComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        trial_pre_questClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "trial_pre_quest"-------
        while continueRoutine:
            # get current time
            t = trial_pre_questClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=trial_pre_questClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *rating* updates
            if rating.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rating.frameNStart = frameN  # exact frame index
                rating.tStart = t  # local t and not account for scr refresh
                rating.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rating, 'tStartRefresh')  # time at next scr refresh
                rating.setAutoDraw(True)
            continueRoutine &= rating.noResponse  # a response ends the trial
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_pre_questComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "trial_pre_quest"-------
        for thisComponent in trial_pre_questComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store data for loop2 (TrialHandler)
        loop2.addData('rating.response', rating.getRating())
        loop2.addData('rating.rt', rating.getRT())
        loop2.addData('rating.started', rating.tStart)
        loop2.addData('rating.stopped', rating.tStop)
        # the Routine "trial_pre_quest" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        loop3 = data.TrialHandler(nReps=1, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions(condFile),
            seed=None, name='loop3')
        thisExp.addLoop(loop3)  # add the loop to the experiment
        thisLoop3 = loop3.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop3.rgb)
        if thisLoop3 != None:
            for paramName in thisLoop3:
                exec('{} = thisLoop3[paramName]'.format(paramName))
        
        for thisLoop3 in loop3:
            currentLoop = loop3
            # abbreviate parameter names if possible (e.g. rgb = thisLoop3.rgb)
            if thisLoop3 != None:
                for paramName in thisLoop3:
                    exec('{} = thisLoop3[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "trial"-------
            continueRoutine = True
            # update component parameters for each repeat
            movie = visual.MovieStim3(
                win=win, name='movie',
                noAudio = False,
                filename=video_ind,
                ori=0, pos=(0, 0), opacity=1,
                loop=False,
                depth=0.0,
                )
            # keep track of which components have finished
            trialComponents = [movie]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "trial"-------
            while continueRoutine:
                # get current time
                t = trialClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=trialClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *movie* updates
                if movie.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    movie.frameNStart = frameN  # exact frame index
                    movie.tStart = t  # local t and not account for scr refresh
                    movie.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(movie, 'tStartRefresh')  # time at next scr refresh
                    movie.setAutoDraw(True)
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "trial"-------
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            movie.stop()
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 1 repeats of 'loop3'
        
        
        # ------Prepare to start Routine "trial_quest"-------
        continueRoutine = True
        # update component parameters for each repeat
        # keep track of which components have finished
        trial_questComponents = [form, button]
        for thisComponent in trial_questComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        trial_questClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "trial_quest"-------
        while continueRoutine:
            # get current time
            t = trial_questClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=trial_questClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *form* updates
            if form.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                form.frameNStart = frameN  # exact frame index
                form.tStart = t  # local t and not account for scr refresh
                form.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(form, 'tStartRefresh')  # time at next scr refresh
                form.setAutoDraw(True)
            
            # *button* updates
            if button.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                button.frameNStart = frameN  # exact frame index
                button.tStart = t  # local t and not account for scr refresh
                button.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button, 'tStartRefresh')  # time at next scr refresh
                button.setAutoDraw(True)
            if button.status == STARTED:
                # check whether button has been pressed
                if button.isClicked:
                    if not button.wasClicked:
                        button.timesOn.append(button.buttonClock.getTime()) # store time of first click
                        button.timesOff.append(button.buttonClock.getTime()) # store time clicked until
                    else:
                        button.timesOff[-1] = button.buttonClock.getTime() # update time clicked until
                    if not button.wasClicked:
                        continueRoutine = False  # end routine when button is clicked
                        None
                    button.wasClicked = True  # if button is still clicked next frame, it is not a new click
                else:
                    button.wasClicked = False  # if button is clicked next frame, it is a new click
            else:
                button.buttonClock.reset() # keep clock at 0 if button hasn't started / has finished
                button.wasClicked = False  # if button is clicked next frame, it is a new click
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_questComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "trial_quest"-------
        for thisComponent in trial_questComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        form.addDataToExp(thisExp, 'rows')
        form.autodraw = False
        loop2.addData('button.started', button.tStartRefresh)
        loop2.addData('button.stopped', button.tStopRefresh)
        loop2.addData('button.numClicks', button.numClicks)
        if button.numClicks:
           loop2.addData('button.timesOn', button.timesOn)
           loop2.addData('button.timesOff', button.timesOff)
        else:
           loop2.addData('button.timesOn', "")
           loop2.addData('button.timesOff', "")
        # the Routine "trial_quest" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1 repeats of 'loop2'
    
    thisExp.nextEntry()
    
# completed 1 repeats of 'loop1'


# ------Prepare to start Routine "text_posthyp"-------
continueRoutine = True
# update component parameters for each repeat
key_resp_4.keys = []
key_resp_4.rt = []
_key_resp_4_allKeys = []
# keep track of which components have finished
text_posthypComponents = [text, key_resp_4]
for thisComponent in text_posthypComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
text_posthypClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "text_posthyp"-------
while continueRoutine:
    # get current time
    t = text_posthypClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=text_posthypClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text* updates
    if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text.frameNStart = frameN  # exact frame index
        text.tStart = t  # local t and not account for scr refresh
        text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
        text.setAutoDraw(True)
    
    # *key_resp_4* updates
    waitOnFlip = False
    if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_4.frameNStart = frameN  # exact frame index
        key_resp_4.tStart = t  # local t and not account for scr refresh
        key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
        key_resp_4.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_4.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_4.getKeys(keyList=None, waitRelease=False)
        _key_resp_4_allKeys.extend(theseKeys)
        if len(_key_resp_4_allKeys):
            key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
            key_resp_4.rt = _key_resp_4_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in text_posthypComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "text_posthyp"-------
for thisComponent in text_posthypComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text.started', text.tStartRefresh)
thisExp.addData('text.stopped', text.tStopRefresh)
# check responses
if key_resp_4.keys in ['', [], None]:  # No response was made
    key_resp_4.keys = None
thisExp.addData('key_resp_4.keys',key_resp_4.keys)
if key_resp_4.keys != None:  # we had a response
    thisExp.addData('key_resp_4.rt', key_resp_4.rt)
thisExp.addData('key_resp_4.started', key_resp_4.tStartRefresh)
thisExp.addData('key_resp_4.stopped', key_resp_4.tStopRefresh)
thisExp.nextEntry()
# the Routine "text_posthyp" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "posthyp_baseline"-------
continueRoutine = True
# update component parameters for each repeat
# keep track of which components have finished
posthyp_baselineComponents = [posthyp]
for thisComponent in posthyp_baselineComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
posthyp_baselineClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "posthyp_baseline"-------
while continueRoutine:
    # get current time
    t = posthyp_baselineClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=posthyp_baselineClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *posthyp* updates
    if posthyp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        posthyp.frameNStart = frameN  # exact frame index
        posthyp.tStart = t  # local t and not account for scr refresh
        posthyp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(posthyp, 'tStartRefresh')  # time at next scr refresh
        posthyp.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in posthyp_baselineComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "posthyp_baseline"-------
for thisComponent in posthyp_baselineComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
posthyp.stop()
# the Routine "posthyp_baseline" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
