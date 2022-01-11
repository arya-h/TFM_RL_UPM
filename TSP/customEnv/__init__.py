from gym.envs.registration import register
register(
     id='truckOpt-v1',
     entry_point='customEnv.envs:truckEnv',
 )
register(
     id='truckOpt-v2',
     entry_point='customEnv.envs:truckEnv',
)