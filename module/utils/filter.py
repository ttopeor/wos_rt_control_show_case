import numpy as np

class DataFilter:
    def __init__(self, config, smooth_factor = 0.1, cov_name = 'fr3_cov'):
        # Initialize offset and filter parameters
        self.initialized = False
        self.offsets = np.zeros(6)
        self.prev_estimate = np.zeros(6)  # Previous state estimate
        # Load measurement covariance from config
        cov = config.get(cov_name)
        if cov is not None:
            self.measurement_covariance = np.array(cov)  # Measurement noise covariance matrix
        else:
            raise ValueError("Config does not contain cov_name key.")

        # Initialize estimate_covariance based on measurement_covariance
        self.estimate_covariance = np.copy(self.measurement_covariance)

        # Set process_covariance to a small value
        self.process_covariance = self.estimate_covariance * smooth_factor 

    def set_offsets(self, offset_array):
        """
        Sets the offsets for the filter.
        
        :param offset_array: Array of length 6 for offsets.
        """
        if len(offset_array) == 6:
            self.offsets = np.array(offset_array)
        else:
            raise ValueError("Offset array must have 6 dimensions.")

    def filter_data(self, input_array):
        """
        Filters the input array by applying an offset and Kalman filter.
        
        :param input_array: 6-dimensional float array to be filtered.
        :return: Filtered array as a 6-dimensional float array.
        """
        if len(input_array) != 6:
            raise ValueError("Input array must have 6 dimensions.")

        # Apply offsets
        adjusted_array = input_array - self.offsets

        if not self.initialized:
            self.prev_estimate = adjusted_array.copy()
            self.initialized = True
            return adjusted_array 
        
        filtered_array = self.kalman_filter(adjusted_array)
        return filtered_array

    def kalman_filter(self, measurement):
        """
        Applies the Kalman filter to smooth the data.
        
        :param measurement: Input measurement array.
        :return: Smoothed estimate.
        """
        # Prediction phase
        predicted_estimate = self.prev_estimate  # No control input
        predicted_covariance = self.estimate_covariance + self.process_covariance

        # Update phase
        innovation = measurement - predicted_estimate
        innovation_covariance = predicted_covariance + self.measurement_covariance

        # Kalman Gain
        kalman_gain = np.dot(predicted_covariance, np.linalg.inv(innovation_covariance))

        # Update estimate with measurement
        current_estimate = predicted_estimate + np.dot(kalman_gain, innovation)

        # Update covariance
        self.estimate_covariance = np.dot((np.eye(6) - kalman_gain), predicted_covariance)

        # Save current estimate for next iteration
        self.prev_estimate = current_estimate

        return current_estimate
