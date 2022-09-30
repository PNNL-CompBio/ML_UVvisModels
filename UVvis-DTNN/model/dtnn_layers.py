import lasagne
import theano
import theano.tensor as T
import numpy as np

class SwitchLayer(lasagne.layers.Layer):
    """
    Layer contains a coefficient matrix.
    Rows from this matrix are returned using the input as row indices.
    The output array thus has an additional dimension.

    Parameters
    ----------
    incoming : :class: `Layer` instances

    num_options : int or T.scalar
        Number of rows in the coefficient matrix

    out_len : int or T.scalar
        Number of columns in coefficient matrix
    """
    def __init__(self, incoming, num_options, out_len, W=lasagne.init.Uniform(1), **kwargs):
        super(SwitchLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.out_len = out_len
        self.C = self.add_param(W, (num_options, self.out_len), name='C')

    def get_output_for(self, input, **kwargs):
        return self.C[input,:]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.out_len)

class MaskLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input>0

class SumMaskedLayer(lasagne.layers.MergeLayer):
    def __init__(self, var, mask, **kwargs):
        super(SumMaskedLayer, self).__init__([var, mask], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][0]

    def get_output_for(self, input, **kwargs):
        var, mask = input
        return T.sum(var*mask, axis=1)

class RecurrentLayer(lasagne.layers.MergeLayer):
    """
        Layer implements the iterative refinement of atom coefficients as
        described in [SCHUTT].

        References
        ----------
        [SCHUTT] Schutt, Kristof T., Farhad Arbabzadah, Stefan Chmiela, Klaus
        R. Muller, and Alexandre Tkatchenko. 2017. "Quantum-Chemical Insights
        from Deep Tensor Neural Networks." Nature Communications 8 (January):
        13890.
    """
    def __init__(self, atomc, dist, atom_mask, num_hidden=60, num_passes=2, include_diagonal=False, nonlinearity=lasagne.nonlinearities.tanh, Wcf=lasagne.init.GlorotNormal(1.0), Wfc=lasagne.init.GlorotNormal(1.0), Wdf=lasagne.init.GlorotNormal(1.0), bcf=lasagne.init.Constant(0.0), bdf=lasagne.init.Constant(0.0), **kwargs):
        super(RecurrentLayer, self).__init__([atomc, dist, atom_mask], **kwargs)
        num_atoms = self.input_shapes[0][1]
        c_len = self.input_shapes[0][2]
        d_len = self.input_shapes[1][3]
        self.Wcf = self.add_param(Wcf, (c_len, num_hidden), name="W_atom_c")
        self.bcf = self.add_param(bcf, (num_hidden, ), name="b_atom_c")
        self.Wdf = self.add_param(Wdf, (d_len, num_hidden), name="W_dist")
        self.bdf = self.add_param(bdf, (num_hidden, ), name="b_dist")
        self.Wfc = self.add_param(Wfc, (num_hidden, c_len), name="W_hidden_to_c")
        self.num_passes = num_passes
        self.nonlin = nonlinearity
        if include_diagonal:
            self.inv_eye_mask = None
        else:
            self.inv_eye_mask = (T.eye(num_atoms,num_atoms) < 1).dimshuffle("x",0,1,"x")


    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        atom_c, dist, atom_mask = input

        c = atom_c

        for i in range(self.num_passes):
            # Contribution from atoms C
            # c has dim (sample, atom_i, feature)
            #   dimshuffle makes it broadcastable as (sample, 1, atom_j, feature)
            term1 = (T.dot(c.dimshuffle(0,"x",1,2), self.Wcf) + self.bcf)
            # Contribution from distances
            # dist has dim (sample, atom_i, atom_j, gaussian_expansion)
            term2 = T.dot(dist, self.Wdf) + self.bdf
            V = self.nonlin(T.dot(term1*term2, self.Wfc))
            # V has dim (sample, atom_i, atom_j, feature)
            # Atom mask zeroes out contribution from missing atoms
            # inv_eye_mask zeroes out contribution from diagonal V_{i,i}
            if self.inv_eye_mask is None:
                masked_V = V*atom_mask.dimshuffle(0,"x",1,"x")
            else:
                masked_V = V*atom_mask.dimshuffle(0,"x",1,"x")*self.inv_eye_mask
            Vsum = T.sum(masked_V, axis=2)
            # Vsum has dim (sample, atom_i, feature)
            c = c + Vsum

        return c


