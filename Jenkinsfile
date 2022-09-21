pipeline {
    agent {
    kubernetes {
      yaml """
kind: Pod
metadata:
  annotations:
spec:
  containers:
  - name: kubectl
    image: 016216169391.dkr.ecr.eu-central-1.amazonaws.com/kubectl:v1.23.3
    imagePullPolicy: Always
    command:
    - cat
    tty: true
  - name: kaniko
    image: gcr.io/kaniko-project/executor:debug
    envFrom:
          - secretRef:
                name: github-token
    resources:
      requests:
        cpu: "2"
        memory: "2Gi"
      limits:
        cpu: "4"
        memory: "7Gi"
    imagePullPolicy: Always
    command:
    - cat
    tty: true
    volumeMounts:
      - name: docker-config
        mountPath: /kaniko/.docker
  volumes:
    - name: docker-config
      configMap:
        name: docker-config
"""
    }
  }
    environment {
        APP="huspy-lama-cleaner-service"
        PHASE=branchToConfig(BRANCH_NAME)
        ENV=environment(BRANCH_NAME)
        AWS_PROFILE="huspy-tools"
        ECR="151712667821.dkr.ecr.eu-central-1.amazonaws.com"
        TIMESTAMP="${sh(script: 'date "+%Y%m%d%H%M%S"', returnStdout: true).trim()}"
        APP_VERSION="${PHASE}_${TIMESTAMP}_v${BUILD_NUMBER}_${GIT_COMMIT}"
    }
    stages {
        stage("Build with kaniko") {
            when {
                anyOf{
                    expression { BRANCH_NAME ==~ /(release.*|fix.*|feature.*|bug.*)/ }
                    buildingTag()
                }
            }
            steps {
                script {
                    env.GIT_COMMIT_MSG = sh (script: 'git log -1 --pretty=%B ${GIT_COMMIT}', returnStdout: true).trim()
                    env.GIT_AUTHOR = sh (script: 'git log -1 --pretty=%ce ${GIT_COMMIT}', returnStdout: true).trim()
                }
                container(name: 'kaniko') {
                sh '''
                /kaniko/executor --cache=true --cache-repo=${ECR}/${APP}/cache --dockerfile `pwd`/Dockerfile --context `pwd` --destination=${ECR}/${APP}:${APP_VERSION}
                '''
                }
            }
            post {
               success {
                   slackSend (channel: "#${slackChannel}", color: '#3380C7', message: "*Lama Cleaner Service*: Image built on <${env.BUILD_URL}|#${env.BUILD_NUMBER}> branch ${env.BRANCH_NAME}")
                   echo 'Compile Stage Successful'
               }
               failure {
                   slackSend (channel: "#${slackChannel}", color: '#F44336', message: "*Lama Cleaner Service*: Image build failed <${env.BUILD_URL}|#${env.BUILD_NUMBER}> branch ${env.BRANCH_NAME}")
                   echo 'Compile Stage Failed'
                   sh "exit 1"
               }
           }
        }

        stage("Deploy") {
            when {
                anyOf{
                expression { BRANCH_NAME ==~ /(release.*|fix.*|feature.*|bug.*)/ }
                buildingTag()
                }
            }
            steps{
                script{
                    container(name: 'kubectl') {
                        sh "kubectl set image deployment ${APP} ${APP}=${ECR}/${APP}:${APP_VERSION} --namespace ${PHASE}"
                        sh "kubectl rollout status deployment ${APP} --namespace ${PHASE}"
                    }
                }
            }
            post {
                success {
                    slackSend (channel: "#${slackChannel}", color: '#4CAF50', message: "*Lama Cleaner Service*: Deployment completed <${env.BUILD_URL}|#${env.BUILD_NUMBER}> commit ${env.GIT_COMMIT[0..6]} branch ${env.BRANCH_NAME}")
                    echo 'Deploy Stage Successful'
                }

                failure {
                    slackSend (channel: "#${slackChannel}", color: '#F44336', message: "*Lama Cleaner Service*: Deployment failed <${env.BUILD_URL}|#${env.BUILD_NUMBER}>")
                    echo 'Deploy Stage Failed'
                    sh "exit 1"
                }
            }
        }

        stage('Anchore Scan') {
            when {
                expression { BRANCH_NAME ==~ /(release.*)/ }
            }
            steps {
                script {
                    writeFile file: 'anchore_images_backend', text: "${ECR}/${APP}:${APP_VERSION}"
                    anchore bailOnFail: false, engineRetries: '1200', name: 'anchore_images_backend'
                }
            }
            post {
               success {
                   slackSend (channel: "#${slackChannel}", color: '#3380C7', message: "*Lama Cleaner Service*: Security scan completed <${env.BUILD_URL}|#${env.BUILD_NUMBER}> branch ${env.BRANCH_NAME}")
                   echo 'Compile Stage Successful'
               }
               failure {
                   slackSend (channel: "#${slackChannel}", color: '#F44336', message: "*Lama Cleaner Service*: Security scan failed <${env.BUILD_URL}|#${env.BUILD_NUMBER}> branch ${env.BRANCH_NAME}")
                   echo 'Compile Stage Failed'
                   sh "exit 1"
               }
           }
        }
    }
}

def branchToConfig(branch) {
     script {
        result = "NULL"
        if (branch ==~ /v\d+\.\d+\.\d+/) {
        result ="production"
        slackChannel = "alerts-prod-deployments"
        echo "BRANCH:${branch} -> CONFIGURATION:${result}"
        }
        if (branch ==~ /release.*/) {
        result = "qa"
        slackChannel = "alerts-dev-deployments"
        echo "BRANCH:${branch} -> CONFIGURATION:${result}"
        }
        if (branch ==~ /feature.*|fix.*|bug.*/ ) {
        properties([
            parameters([
                choice(choices: ['','sandbox','QA'], description: 'Select an Environment to Build on it', name: 'ENV_NAME')
            ])
        ])
        result = params.ENV_NAME
        if(params.ENV_NAME == '' ){
            error "Please select an environment..."
        }
        result = "sandbox"
        slackChannel = "alerts-dev-deployments"
        }
        return result
}
}