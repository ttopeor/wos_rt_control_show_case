import time
import threading
import logging
from wos_api.connection import CreateWSClient
from utils.config_loader import load_config


class ReadArmPositionLoop:
    def __init__(self, robot_id,wos_endpoint):
        """
        Initialize the ReadPositionLoop by loading configuration and establishing a connection to the robot.

        """

        self.resource = robot_id
        self.client = CreateWSClient(wos_endpoint)
        success = self.client.connect()
        if not success:
            logging.error("Connection Failed, quitting.")
            exit(1)  # Exit if connection fails

        # Lock to ensure thread-safe access to position_result
        self.position_lock = threading.Lock()

        # Event to control the spinning loop
        self._stop_event = threading.Event()
        self._thread = None

        # Initialize position_result with default values
        position_result, position_err = self.client.run_request(self.resource, "get-cartesian-state", {})

        if position_err:
            logging.error(f"Error getting init position data: {position_err}")
        else:
            with self.position_lock:
                self.position_result = position_result["position"]


    def loop(self):
        """
        Execute the loop task: send a request to get cartesian state data and update position_result.
        """
        position_result, position_err = self.client.run_request(self.resource, "get-cartesian-state", {})

        if position_err:
            logging.error(f"Error getting position data: {position_err}")
        else:
            with self.position_lock:
                self.position_result = position_result["position"]
            # Optionally, log the position data
            #logging.info(f"Position data received: {self.position_result}")

    def get_position_reading(self):
        """
        Get the latest position reading.

        :return: List of position values or [None, ...] if an error occurred.
        """
        with self.position_lock:
            return self.position_result.copy()

    def _spin_loop(self, frequency):
        """
        Internal method to run the spinning loop in a separate thread.

        :param frequency: Loop frequency in Hz.
        """
        interval = 1.0 / frequency  # Seconds
        logging.info(f"ReadPositionLoop thread started at {frequency} Hz (every {interval * 1000:.2f} ms)")
        while not self._stop_event.is_set():
            # Execute the loop task
            self.loop()

            # Sleep for the interval duration
            time.sleep(interval)

    def loop_spin(self, frequency=100):
        """
        Start the spinning loop in a separate thread that runs the loop method at the specified frequency.

        :param frequency: Loop frequency in Hz. Default is 100 Hz.
        """
        if self._thread and self._thread.is_alive():
            logging.warning("ReadPositionLoop is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin_loop, args=(frequency,), daemon=True)
        self._thread.start()
        logging.info(f"ReadPositionLoop spinning thread has been started at {frequency} Hz.")

    def stop_spin(self):
        """
        Stop the spinning loop gracefully.
        """
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
            logging.info("ReadPositionLoop spinning thread has been stopped.")
            self._thread = None
        else:
            logging.warning("ReadPositionLoop spinning thread is not running.")

    def shutdown(self):
        """
        Shutdown the spinning loop and close the client connection.
        """
        self.stop_spin()
        # Ensure that the client connection is properly closed
        if hasattr(self.client, 'close') and callable(getattr(self.client, 'close')):
            self.client.close()
            logging.info("ReadPositionLoop client connection has been closed.")
        else:
            logging.warning("ReadPositionLoop client does not have a 'close' method. Resources may not be released properly.")


if __name__ == "__main__":
    """
    Example usage of ReadPositionLoop with threading.
    This block can be used for testing purposes.
    """
    # Configure logging at the beginning of your script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        config = load_config()
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        exit(1)  # Exit if configuration is missing

    # Initialize the ReadPositionLoop
    read_position_loop = ReadArmPositionLoop(config)

    # Start the spinning loop
    read_position_loop.loop_spin(frequency=100)

    try:
        while True:
            # Example: Access position data every second
            position_data = read_position_loop.get_position_reading()
            logging.info(f"Latest Position Data: {position_data}")
            time.sleep(1)  # Sleep for 1 second
    except KeyboardInterrupt:
        logging.info("\nReadPositionLoop stopped by user.")
    finally:
        # Shutdown the loop and clean up resources
        read_position_loop.shutdown()