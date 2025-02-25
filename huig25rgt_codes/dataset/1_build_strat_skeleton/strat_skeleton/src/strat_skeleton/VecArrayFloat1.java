/****************************************************************************
Copyright (c) 2009, Colorado School of Mines and others. All rights reserved.
This program and accompanying materials are made available under the terms of
the Common Public License - v1.0, which accompanies this distribution, and is 
available at http://www.eclipse.org/legal/cpl-v10.html
****************************************************************************/
package strat_skeleton;

/**
 * A vector represented by a 1D array[n1] of floats.
 * @author Dave Hale, Colorado School of Mines
 * @version 2009.09.15
 */
public class VecArrayFloat1 implements Vec {

  /**
   * Constructs a zero vector with specified dimensions.
   * @param n1 the number of floats in the 1st dimension.
   */
  public VecArrayFloat1(int n1) {
    _a = new float[n1];
    _n1 = n1;
  }

  /**
   * Constructs a vector that wraps the specified array of floats.
   * @param a the array of floats; by reference, not by copy.
   */
  public VecArrayFloat1(float[] a) {
    _a = a;
    _n1 = a.length;
  }

  /**
   * Gets the array of floats wrapped by this vector.
   * @return the array of floats; by reference, not by copy.
   */
  public float[] getArray() {
    return _a;
  }

  /**
   * Gets the number of floats in the 1st array dimension.
   * @return the number of floats in the 1st dimension.
   */
  public int getN1() {
    return _n1;
  }

  public double epsilon() {
    return Math.ulp(1.0f);
  }

  public VecArrayFloat1 clone() {
    VecArrayFloat1 v = new VecArrayFloat1(_n1);
    System.arraycopy(_a,0,v._a,0,_n1);
    return v;
  }

  public double dot(Vec vthat) {
    float[] athis = _a;
    float[] athat = ((VecArrayFloat1)vthat)._a;
    double sum = 0.0;
    for (int i1=0; i1<_n1; ++i1)
      sum += athis[i1]*athat[i1];
    return sum;
  }

  public double norm2() {
    double sum = 0.0;
    for (int i1=0; i1<_n1; ++i1) {
      double ai = _a[i1];
      sum += ai*ai;
    }
    return Math.sqrt(sum);
  }

  public void zero() {
    for (int i1=0; i1<_n1; ++i1)
      _a[i1] = 0.0f;
  }

  public void scale(double s) {
    for (int i1=0; i1<_n1; ++i1)
      _a[i1] *= s;
  }

  public void add(double sthis, Vec vthat, double sthat) {
    float[] athis = _a;
    float[] athat = ((VecArrayFloat1)vthat)._a;
    float fthis = (float)sthis;
    float fthat = (float)sthat;
    for (int i1=0; i1<_n1; ++i1)
      athis[i1] = athis[i1]*fthis+athat[i1]*fthat;
  }

  private float[] _a;
  private int _n1;
}
