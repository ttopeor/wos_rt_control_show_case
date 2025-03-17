# ReadForceLoop.py

import time
import threading
import logging
from wos_api.connection import CreateWSClient
from utils.config_loader import load_config


class ReadForceLoop:
    def __init__(self, robot_id,wos_endpoint):
        """
        Initialize the ReadForceLoop by loading configuration and establishing a connection to the robot.

        :param config: Configuration dictionary containing 'robot_resource_id' and 'wos_end_point'.
        """
        self.resource = robot_id
        self.client = CreateWSClient(wos_endpoint)
        success = self.client.connect()
        if not success:
            logging.error("Connection Failed, quitting.")
            exit(1)  # Exit if connection fails

        # Initialize force_result with default values
        self.force_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Lock to ensure thread-safe access to force_result
        self.force_lock = threading.Lock()

        # Event to control the spinning loop
        self._stop_event = threading.Event()
        self._thread = None

    def loop(self):
        """
        Execute the loop task: send a request to get end-effector force data and update force_result.
        """
        try:
            force_result, force_err = self.client.run_request(self.resource, "get-ee-force", {})

            if force_err:
                logging.error(f"Error getting force data: {force_err}")
            else:
                with self.force_lock:
                    self.force_result = force_result
                # Optionally, log the force data
                #logging.info(f"Force data received: {self.force_result}")

        except Exception as e:
            logging.error(f"Exception in ReadForceLoop: {e}")
            with self.force_lock:
                self.force_result = [None] * 6  # Reset to None or handle as needed

    def get_force_reading(self):
        """
        Get the latest force reading.

        :return: List of force values or [None, ...] if an error occurred.
        """
        with self.force_lock:
            return self.force_result.copy()

    def _spin_loop(self, frequency):
        """
        Internal method to run the spinning loop in a separate thread.

        :param frequency: Loop frequency in Hz.
        """
        interval = 1.0 / frequency  # Seconds
        logging.info(f"ReadForceLoop thread started at {frequency} Hz (every {interval * 1000:.2f} ms)")
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
            logging.warning("ReadForceLoop is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin_loop, args=(frequency,), daemon=True)
        self._thread.start()
        logging.info(f"ReadForceLoop spinning thread has been started at {frequency} Hz.")

    def stop_spin(self):
        """
        Stop the spinning loop gracefully.
        """
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
            logging.info("ReadForceLoop spinning thread has been stopped.")
            self._thread = None
        else:
            logging.warning("ReadForceLoop spinning thread is not running.")

    def shutdown(self):
        """
        Shutdown the spinning loop and close the client connection.
        """
        self.stop_spin()
        # Ensure that the client connection is properly closed
        if hasattr(self.client, 'close') and callable(getattr(self.client, 'close')):
            self.client.close()
            logging.info("ReadForceLoop client connection has been closed.")
        else:
            logging.warning("ReadForceLoop client does not have a 'close' method. Resources may not be released properly.")


if __name__ == "__main__":
    """
    Example usage of ReadForceLoop with threading.
    This block can be used for testing purposes.
    """
    # Configure logging at the beginning of your script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        config = load_config()
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        exit(1)  # Exit if configuration is missing

    # Initialize the ReadForceLoop
    read_force_loop = ReadForceLoop(config)

    # Start the spinning loop
    read_force_loop.loop_spin(frequency=100)

    try:
        while True:
            # Example: Access force data every second
            force_data = read_force_loop.get_force_reading()
            logging.info(f"Latest Force Data: {force_data}")
            time.sleep(1)  # Sleep for 1 second
    except KeyboardInterrupt:
        logging.info("\nReadForceLoop stopped by user.")
    finally:
        # Shutdown the loop and clean up resources
        read_force_loop.shutdown()
