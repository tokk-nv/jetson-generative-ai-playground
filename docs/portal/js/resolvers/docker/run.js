import { substitution } from "../../nanolab";

/*
 * Generate 'docker run' templates for launching models & containers
 */
export function docker_run(env) {
  
  //console.group(`[GraphDB]  Generating docker_run for ${env.key}`);
  //console.log(env);

  const opt = wrapLines(
    nonempty(env.docker_options) ? env.docker_options : docker_options(env)
  ) + ' \\\n ';

  const image = `${env.docker_image} \\\n   `; 
  const exec = `${env.docker_cmd} \\\n   `;
   
  let args = docker_args(env);
  let cmd = nonempty(env.docker_run) ? env.docker_run : env.db.index['docker_run'].value;

  cmd = cmd
    .trim()
    .replace('$OPTIONS', '${OPTIONS}')
    .replace('$IMAGE', '${IMAGE}')
    .replace('$COMMAND', '${COMMAND}')
    .replace('$ARGS', '${ARGS}');

  if( !cmd.endsWith('${ARGS}') )
    args += ` \\\n      `;  // line break for user args

  cmd = cmd
    .replace('${OPTIONS}', opt)
    .replace('${IMAGE}', image)
    .replace('${COMMAND}', exec)
    .replace('${ARGS}', args);

  cmd = substitution(cmd, {
    'MODEL': get_model_repo(env.url ?? env.model_name)
  });

  cmd = cmd
    .replace('\\ ', '\\')
    .replace('  \\', ' \\');  

  //console.log(`[GraphDB] `, cmd);
  //console.groupEnd();

  if( cmd.endsWith(' \\') )
    cmd = cmd.slice(0, -2);

  if( cmd.endsWith('\\') )
    cmd = cmd.slice(0, -1);

  return cmd; 
}

Resolver({
  func: docker_run,
  name: 'Docker Run Cmd',
  title: 'Docker Run',
  filename: 'docker-run.sh',
  value: "docker run $OPTIONS $IMAGE $COMMAND $ARGS",
  group: 'shell',
  tags: ['string', 'shell'],
  help: [
    `Template that builds the 'docker run' command from $OPTIONS $IMAGE $COMMAND $ARGS\n`,
    `You can more deeply customize the container settings by altering these.`,
  ],
  text: `Run these terminal commands from the host device or SSH, this one downloading the model and starting an <span class="code">openai.chat.completion</span> server:`,
  footer: `These individual commands are typically meant for exploratory use - see the <span class="code">Compose</span> tab for managed deployments of models and microservices.`
});