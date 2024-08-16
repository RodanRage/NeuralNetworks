using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    internal class MatrixOperations
    {
         public static float[] GetVectMatMul(float[] vector, float[][] matrix)
        {
            float[] outputVector = new float[] { 0, 0, 0 };
            if (vector.Length != matrix.Length)
            {
                throw new Exception("Размерность массивов не совпадает");
            }
            else
            {
                for (int i = 0; i < vector.Length; i++)
                {
                    outputVector[i] = GetWsum(vector, matrix[i]);
                }
                return outputVector;
            }
        }

        public static float GetWsum(float[] firstVector, float[] secondVector)
        {
            if (secondVector.Length != firstVector.Length)
            {
                throw new Exception("Размерность массивов не совпадает");
            }
            else
            {
                float output = 0;
                for (int i = 0; i < firstVector.Length; i++)
                {
                    output += firstVector[i] * secondVector[i];
                }
                return output;
            }
        }

        public static float GetWsum(byte[] firstVector, float[] secondVector)
        {
            if (secondVector.Length != firstVector.Length)
            {
                throw new Exception("Размерность массивов не совпадает");
            }
            else
            {
                float output = 0;
                for (int i = 0; i < firstVector.Length; i++)
                {
                    output += firstVector[i] * secondVector[i];
                }
                return output;
            }
        }

        public static float[] GetEleMul(float scalar, float[] vector)
        {
            float[] output = new float[] { 0, 0, 0 };
            if (vector.Length == output.Length)
            {
                for (int i = 0; i < vector.Length; i++)
                {
                    output[i] = vector[i]*scalar;
                }
                return output;
            }
            throw new Exception("Несоответствие размерностей входного и выходного массивов");
        }

        public static float[][] GetOuterProd(float[] firstVector, float[] secondVector)
        {
            float[][] outputMatrix = new float[firstVector.Length][];
            for(int i = 0; i < firstVector.Length; i++)
            {
                outputMatrix[i] = new float[secondVector.Length];
                for (int j = 0; j < secondVector.Length; j++)
                {
                    outputMatrix[i][j] = firstVector[j] * secondVector[i];
                }
            }
            return outputMatrix;
        }
    }
}
