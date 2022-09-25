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
      - name: ecr-profile
        mountPath: /root/.aws
  volumes:
    - name: docker-config
      configMap:
        name: docker-config
    - name: ecr-profile
      configMap:
        name: ecr-profile
"""
    }
  }
    environment {
        APP="huspy-lama-cleaner-service"
        ENV=environment(BRANCH_NAME)
        AWS_PROFILE="huspy-tools"
        ECR="151712667821.dkr.ecr.eu-central-1.amazonaws.com"
        APP_VERSION="${ENV}_v${BUILD_NUMBER}_${GIT_COMMIT}"
    }

    stages {
        stage("Build with Kaniko") {
            when {
              anyOf {
               expression { BRANCH_NAME ==~ /(release.*|development|fix.*|feature.*|bug.*|main)/ }
              }
            }
            steps {
                container(name: 'kaniko') {
                sh """
                /kaniko/executor --dockerfile `pwd`/Dockerfile --context `pwd` --destination=${ECR}/${APP}:${APP_VERSION}
                """
                }
            }
        }

        stage("Update image in argocd") {
            when {
              anyOf {
                expression { BRANCH_NAME ==~ /(release.*|development|fix.*|feature.*|bug.*|main)/ }
              }
            }
            steps {
                container(name: 'alpine') {
                  dir('huspy-services') {
                    git branch: 'main', credentialsId: 'yahiakhidr', url: 'https://github.com/huspy/huspy-services.git'
                  }
                  sh """
                  apk add yq git
                  cd huspy-services
                  yq e -i '.microservice.image.tag = "${APP_VERSION}"' helm/${APP}/${ENV}.yaml
                  git config --global --add safe.directory ${WORKSPACE}/huspy-services
                  git config --global user.email "github_actor@users.noreply.github.com"
                  git config --global user.name "github_actor"
                  git checkout main
                  git add helm/${APP}/${ENV}.yaml
                  git commit -m "Update ${APP} image in ${ENV}"
                  """
                  withCredentials([gitUsernamePassword(credentialsId: 'yahiakhidr')]) {
                    dir('huspy-services') {
                      sh 'git push origin main'
                    }
                  }
                }
            }
        }
    }
}

def environment(branch) {
  script {
    if (branch ==~ /main|v\d+\.\d+\.\d+/) {
      return "prod"
    }
    if (branch ==~ /feature.*|development|fix.*|bug.*/ ) {
      return "dev"
    }
    if (branch ==~ /release.*/ ) {
      return "qa"
    }
  }
}

def branchToConfig(branch) {
  script {
    result = "NULL"
    if (branch == 'main') {
      result = "production"
      slackChannel = "alerts-prod-deployments"
      echo "BRANCH:${branch} -> CONFIGURATION:${result}"
    }
    if (branch ==~ /feature.*|development|fix.*|bug.*/) {
      result = "development"
      slackChannel = "alerts-prod-deployments"
      echo "BRANCH:${branch} -> CONFIGURATION:${result}"
    }
    if (branch ==~ /release.*/) {
      result = "qa"
      slackChannel = "alerts-prod-deployments"
      echo "BRANCH:${branch} -> CONFIGURATION:${result}"
    }
    return result
  }
}