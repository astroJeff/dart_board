import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model('Neural_Net_Parameters_to_Photometry')

def add_columns(array, dtype_additions):
    """
    Adds new empty columns to a numpy array with determined
    column names and types

    Parameters:
              array: a numpy array with specific column names and types
              dtype_additions: a list containing tuples that describe each
                               new column by its name and type

    Returns:
           new_output: a numpy array consisting of the original array and the
                       newly added column types
    """

    new_dtype = np.dtype(array.dtype.descr + dtype_additions)
    new_output = np.zeros(len(array), dtype=new_dtype)

    for col in array.dtype.names:
        new_output[col] = array[col]

    return new_output


def model_predictions(mass, gravity, Teff, FeH):

    """
    Calls the pretrained neural network that predicts the color indices B-V, U-B
    and the absolute V magnitude of a star using its mass, gravity, Temperature 
    and metallicity

    Parameters:
              mass: the mass of the star in solar masses
              gravity: the surface gravity of the star in cgs units
              Teff: the effective temperature of the star in Kelvins
              FeH: the metallicity of the star in terms of its logarithmic
                   ratio to the solar metallicity
    
    Returns:
           phot: a numpy array containing the predicted B-V, U-B indices and
                 the absolute V magnitude 
    """
    Input = np.array([Teff, gravity, mass, FeH]).reshape(1,4)
    phot = model.predict(Input)
    return phot




def calc_photometry(output, Z):
    """
    Takes an array with stellar parameters and the metallicity of the binary
    system, calculates the photometry of the two components and appends it in
    the array

    Parameters:
              output: a numpy array containing stellar parameters of the 2
              components of the system
              Z: the metallicity of the system

    Returns:
           new_output: a numpy array consisting of the initial parameters of the
                       binary and the photometric quantities of each of the two
                       components
    """

    new_output = add_columns(output, [('B1','f8'), ('V1','f8'), ('U1','f8'), ('B2','f8'), ('V2','f8'), ('U2','f8')])

    if new_output['k1'][0] < 10:

        mass1 = new_output['M1'][0]
        gravity1 = (mass1 / new_output['R1'][0]**2)*27400
        Teff1 = new_output['Teff1'][0]

        B_V1, U_B1, V1 = model_predictions(mass1, gravity1, Teff1, Z)[0]
        B1 = B_V1 + V1
        U1 = U_B1 + B1
        

        new_output['U1'][0] = U1
        new_output['V1'][0] = V1
        new_output['B1'][0] = B1
    else:
        new_output['U1'][0] = 99
        new_output['V1'][0] = 99
        new_output['B1'][0] = 99

    if new_output['k2'][0] < 10:

        mass2 = new_output['M2'][0]
        gravity2 = (mass2 / new_output['R2'][0]**2)*27400
        Teff2 = new_output['Teff2'][0]

        B_V2, U_B2, V2 = model_predictions(mass2, gravity2, Teff2, Z)[0]
        B2 = B_V2 + V2
        U2 = U_B2 + B2
        

        new_output['U2'][0] = U2
        new_output['V2'][0] = V2
        new_output['B2'][0] = B2
    else:
        new_output['U2'][0] = 99
        new_output['V2'][0] = 99
        new_output['B2'][0] = 99


    return new_output
