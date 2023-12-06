import logging

log_dir = './logs'
file_name = 'DQN.log'

logging.basicConfig(level=logging.DEBUG, filename=f'{log_dir}/{file_name}', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
# Create a parent logger
parent_logger = logging.getLogger('parent')

# Create child loggers
child1_logger = logging.getLogger('parent.child1')
child2_logger = logging.getLogger('parent.child2')

# Log messages
parent_logger.debug('This is a debug message from parent')
child1_logger.info('This is an info message from child1')
child2_logger.warning('This is a warning message from child2')