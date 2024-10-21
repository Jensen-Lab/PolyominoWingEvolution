import numpy as np
import random
from scipy import ndimage
from scipy.ndimage import label

## Insects in experiments have aspect ratio 5 and wingsize of 9 elements

## CHECK WHETHER THE INSECT IS VALID

def insectCheck(insect,checkValues,aspectRatio):

        # adding a boundary outside the insect
        borderSide = np.zeros((aspectRatio,1))
        borderTop = np.zeros(2*aspectRatio + 3)
        insect = np.concatenate((np.concatenate((borderSide,insect),axis = 1),borderSide),axis = 1)
        insect = np.vstack((np.vstack((borderTop,insect)),borderTop))

        # check for empty spaces inside body
        _, featureNo = label(insect)

        # check for empty spaces inside body
        _, negFeatureNo = label(1 - insect)
        
        return featureNo == checkValues[0] and negFeatureNo == checkValues[1]

## GENERATING A DAUGHTER BASED ON THE PROBABILITY MAP

def daughterGen_prob(parent1,parent2,wingsize,aspectRatio):
    
    # initiate for check of simply rook-wise connected polyominoes condition
    result = False
    checkValues = np.array([1,1])
    
    # insect body
    body = np.ones((aspectRatio,1))

    # order:
    # calculate the parent wing probability map
    # keep all the elements where the value is 1 (overlap between parents)
    # determine all indices where the value is 0.5 (i.e. one parent has a wing element present)
    # pick n random indices with a present wing element in a parent and fill them with ones, where n is the number of desired elements minus the number of existing ones
    # create final insect
    # check that the insect is valid

    # calculate parent wing probability map - map of overlapping elements
    probability = (parent1 + parent2)/2
    probWing = np.copy(probability[:,0:aspectRatio])
    
    # run until a simply rook-wise connected lattice animal is created
    while result == False:

        # keep all the elements where the value is 1 (overlap between parents)
        wing = np.where(probWing < 1, 0, probWing)
        
        # determine all indices where the value is 0.5 (i.e. one parent has a wing element present)
        idx05 = np.argwhere(probWing == 0.5)

        # pick n random indices with a present wing element in a parent and fill them with ones, where n is the number of desired elements minus the number of existing ones
        n = int(wingsize - np.sum(wing))
        idxs = random.sample(range(len(idx05)), n)
        wing[tuple(idx05[idxs].T)] = 1

        # create the final insect
        insect = np.concatenate((np.concatenate((wing,body),axis = 1),np.fliplr(wing)),axis = 1)

        # check validity
        result = insectCheck(insect,checkValues,aspectRatio)
    
    # return the final insect
    return insect

## GENERATING A MUTATION THAT SWAPS OUT ONE ELEMENT AND PLACES IT IN AN EMPTY ELEMENT
def mutateDaughter(daughter,aspectRatio):

    # initiate for check of simply rook-wise connected polyominoes condition
    result = False
    checkValues = np.array([1,1])
    
    # insect body
    body = np.ones((aspectRatio,1))

    # order:
    # extract the daughter wing
    # determine all indices where the value is 1 and 0
    # choose one randome index in both sets and switch the value
    # create final insect
    # check that the insect is valid

    # run until a simply rook-wise connected lattice animal is created
    while result == False:

        # extract the daughter wing
        wing = np.copy(daughter[:,0:aspectRatio])

        # determine all indices where the value is 1 and 0
        idx0 = np.argwhere(wing == 0)
        idx1 = np.argwhere(wing == 1)

        # choose one randome index in both sets and switch the value
        idx0s = random.sample(range(len(idx0)), 1)
        wing[tuple(idx0[idx0s].T)] = 1
        idx1s = random.sample(range(len(idx1)), 1)
        wing[tuple(idx1[idx1s].T)] = 0

        # create final insect
        insect = np.concatenate((np.concatenate((wing,body),axis = 1),np.fliplr(wing)),axis = 1)

        # check validity
        result = insectCheck(insect,checkValues,aspectRatio)

    return insect

## FUNCTION THAT ENCOMPASSES ALL FUNCTIONS

def daughterGen(parent1,parent2,aspectRatio,wingsize,mutateProbability):

    # create breeded daughter
    insect = daughterGen_prob(parent1,parent2,wingsize,aspectRatio)
    
    # Mutate wing with the probability given
    if random.random() < mutateProbability:
        #print('Daughter is mutated')
        insect = mutateDaughter(insect,aspectRatio)

    return insect