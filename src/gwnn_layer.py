"""GWNN layers."""

import torch
from torch_sparse import spspmm, spmm


class GraphWaveletLayer(torch.nn.Module):
    """
    Abstract Graph Wavelet Layer class.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param ncount: Number of nodes.
    :param device: Device to train on.
    """

    def __init__(self, in_channels, out_channels, ncount, device):
        super(GraphWaveletLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncount = ncount
        self.device = device
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining diagonal filter matrix (Theta in the paper) and weight matrix.
        """
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.in_channels, self.out_channels))
        self.diagonal_weight_indices = torch.LongTensor([[node for node in range(self.ncount)],
                                                         [node for node in range(self.ncount)]])

        self.diagonal_weight_indices = self.diagonal_weight_indices.to(
            self.device)
        self.diagonal_weight_filter = torch.nn.Parameter(
            torch.Tensor(self.ncount, 1))

    def init_parameters(self):
        """
        Initializing the diagonal filter and the weight matrix.
        """
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.9, 1.1)
        torch.nn.init.xavier_uniform_(self.weight_matrix)


class SparseGraphWaveletLayer(GraphWaveletLayer):
    """
    Sparse Graph Wavelet Layer Class.
    """

    def forward(self, phi_indices_list, phi_values_list, phi_inverse_indices_list,
                phi_inverse_values_list, feature_indices, feature_values, dropout):
        """
        Forward propagation pass.
        :param phi_indices_list: A list of sparse wavelet matrix index pairs.
        :param phi_values_list: A list of sparse wavelet matrix values.
        :param phi_inverse_indices_list: A list of inverse wavelet matrix index pairs.
        :param phi_inverse_values_list: A list of inverse wavelet matrix values.
        :param feature_indices: Feature matrix index pairs.
        :param feature_values: Feature matrix values.
        :param dropout: Dropout rate.
        :return dropout_features: Filtered feature matrix extracted.
        """
        filtered_features = spmm(feature_indices,
                                 feature_values,
                                 self.ncount,
                                 self.in_channels,
                                 self.weight_matrix)

        localized_features_list = torch.empty(
            len(phi_indices_list), self.ncount, self.out_channels, dtype=float)

        for i, _ in enumerate(phi_indices_list):
            rescaled_phi_indices, rescaled_phi_values = spspmm(phi_indices_list[i],
                                                               phi_values_list[i],
                                                               self.diagonal_weight_indices,
                                                               self.diagonal_weight_filter.view(
                                                               -1),
                                                               self.ncount,
                                                               self.ncount,
                                                               self.ncount)

            phi_product_indices, phi_product_values = spspmm(rescaled_phi_indices,
                                                             rescaled_phi_values,
                                                             phi_inverse_indices_list[i],
                                                             phi_inverse_values_list[i],
                                                             self.ncount,
                                                             self.ncount,
                                                             self.ncount)
            localized_features = spmm(phi_product_indices,
                                      phi_product_values,
                                      self.ncount,
                                      self.ncount,
                                      filtered_features)

            localized_features_list[i] = localized_features

        dropout_features = torch.nn.functional.dropout(torch.nn.functional.relu(localized_features_list),
                                                       training=self.training,
                                                       p=dropout)
        return dropout_features


class DenseGraphWaveletLayer(GraphWaveletLayer):
    """
    Dense Graph Wavelet Layer Class.
    """

    def forward(self, phi_indices_list, phi_values_list, phi_inverse_indices_list, phi_inverse_values_list, features):
        """
        Forward propagation pass.
        :param phi_indices_list: A lsit of sparse wavelet matrix index pairs.
        :param phi_values_list: A lsit of sparse wavelet matrix values.
        :param phi_inverse_indices_list: A list of inverse wavelet matrix index pairs.
        :param phi_inverse_values_list: A list of inverse wavelet matrix values.
        :param features: Feature matrix.
        :return localized_features: Filtered feature matrix extracted.
        """

        localized_features_list = torch.empty(
            len(phi_indices_list), self.ncount, self.out_channels, dtype=float)

        for i, _ in enumerate(phi_indices_list):
            rescaled_phi_indices, rescaled_phi_values = spspmm(phi_indices_list[i],
                                                               phi_values_list[i],
                                                               self.diagonal_weight_indices,
                                                               self.diagonal_weight_filter.view(
                                                                   -1),
                                                               self.ncount,
                                                               self.ncount,
                                                               self.ncount)

            phi_product_indices, phi_product_values = spspmm(rescaled_phi_indices,
                                                             rescaled_phi_values,
                                                             phi_inverse_indices_list[i],
                                                             phi_inverse_values_list[i],
                                                             self.ncount,
                                                             self.ncount,
                                                             self.ncount)

            filtered_features = torch.mm(
                features[i], self.weight_matrix.double())

            localized_features = spmm(phi_product_indices,
                                      phi_product_values,
                                      self.ncount,
                                      self.ncount,
                                      filtered_features)

            localized_features_list[i] = localized_features

        return localized_features_list
