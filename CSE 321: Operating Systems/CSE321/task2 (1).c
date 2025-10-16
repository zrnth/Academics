#include <stdio.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <string.h>
#include <sys/wait.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

struct message {
    long int messageType;
    char content[200];
};

int main() {
    pid_t childProcess;
    int messageQueueID;
    int sendStatus;
    struct message msgData;
    char inputBuffer[200];
    key_t queueKey = ftok("task2.c", 'A');

    if (queueKey == -1) {
        perror("ftok");
        exit(1);
    }

    messageQueueID = msgget(queueKey, 0666 | IPC_CREAT);
    if (messageQueueID == -1) {
        perror("msgget");
        exit(1);
    }

    printf("Please enter the workspace name:\n");
    read(0, inputBuffer, sizeof(inputBuffer));

    childProcess = fork();
    if (childProcess > 0) {
        if (strncmp(inputBuffer, "cse321", 6) != 0) {
            printf("Invalid workspace name\n");
            kill(0, SIGTERM);
            exit(1);
        }

        strcpy(msgData.content, inputBuffer);
        msgData.messageType = 2;
        sendStatus = msgsnd(messageQueueID, (void *)&msgData, sizeof(msgData.content), 0);
        printf("Workspace name sent to OTP generator from login: %s\n", msgData.content);

        if (sendStatus == -1) {
            perror("msgsnd");
            exit(1);
        }

        wait(NULL);
        struct message receivedMsg;
        msgrcv(messageQueueID, (void *)&receivedMsg, sizeof(receivedMsg.content), 0, IPC_NOWAIT);
        printf("Login received OTP from OTP generator: %s\n", receivedMsg.content);

        struct message mailMsg;
        msgrcv(messageQueueID, (void *)&mailMsg, sizeof(mailMsg.content), 0, IPC_NOWAIT);
        printf("Login received OTP from Mail: %s\n", mailMsg.content);

        // Compare OTPs and print verification result
        if (strcmp(receivedMsg.content, mailMsg.content) == 0) {
            printf("OTP Verified\n");
        } else {
            printf("OTP Mismatch\n");
        }

        msgctl(messageQueueID, IPC_RMID, 0);

    } else if (childProcess == 0) {
        int otpType = 2;
        struct message receivedMessage;
        int receiveStatus;
        char otpBuffer[200];

        msgrcv(messageQueueID, (void *)&receivedMessage, sizeof(receivedMessage.content), otpType, IPC_NOWAIT);
        printf("OTP generator received workspace name from login: %s\n", receivedMessage.content);

        pid_t otpGenPID = getpid();
        sprintf(otpBuffer, "%d", otpGenPID);
        strcpy(receivedMessage.content, otpBuffer);
        receivedMessage.messageType = 1;
        sendStatus = msgsnd(messageQueueID, (void *)&receivedMessage, sizeof(receivedMessage.content), 0);
        printf("OTP sent to login from OTP generator: %s\n", receivedMessage.content);

        pid_t mailProcess = fork();
        if (mailProcess > 0) {
            receivedMessage.messageType = 3;
            sendStatus = msgsnd(messageQueueID, (void *)&receivedMessage, sizeof(receivedMessage.content), 0);
            printf("OTP sent to mail from OTP generator: %s\n", receivedMessage.content);
            wait(NULL);
        } else if (mailProcess == 0) {
            int mailType = 3;
            struct message mailData;
            msgrcv(messageQueueID, (void *)&mailData, sizeof(mailData.content), mailType, IPC_NOWAIT);
            printf("Mail received OTP from OTP generator: %s\n", mailData.content);
            mailData.messageType = 4;
            sendStatus = msgsnd(messageQueueID, (void *)&mailData, sizeof(mailData.content), 0);
        }
    }

    return 0;
}
