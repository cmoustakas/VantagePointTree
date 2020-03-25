/*
The MIT License (MIT)
Copyright (c) 2014
Athanassios Kintsakis
Contact
athanassios.kintsakis@gmail.com
akintsakis@issel.ee.auth.gr
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include  <time.h>
#include <sys/time.h>
#include<stdbool.h>
#include<unistd.h>

#define LEFT -1
#define RIGHT 1
#define LOWER_TAG   2
#define HIGHER_TAG  3
#define KNN_TAG 4

MPI_Status Stat;
MPI_Comm sub_comm;

void partition (int *array, int elements, int pivot, int **arraysmall, int **arraybig, int *endsmall, int *endbig);
int selection(int *array,int number);

/** MY FUNCTIONS : **/
int *selectMasters(int level,int noProc);
int nearestLeftMasterNode(int*masterNodesArray,int processId,int masterArraylen);
void partitionAndSendArrays(int processId,int noProcesses,int sumMax,int sumEq,int sumMin,int median,int partLength,int size,int* recvBuff,int* numberPart);
void allKnnAlgo(int processId,int* numberPart,int partLength,int boundsOfk,int* keepMedian,int level);
void findNN(int brother,int processId,int*numberPart,int partLength,int*keepMedian,int boundsOfk,int stdPoint,int direction,int level);
int findMinDistance(int *numberPart,int partLength,int previousDist,int stdPoint,int median);
/***Kills processes that have no values left in their arrays****/
void removeElement(int *array, int *size, int element)
{
    int i;
    int flag=0;
    for(i=0; i<*size; i++)
    {
        if(flag==1)
            array[i]=array[i+1];
        if(array[i]==element&& flag==0)
        {
            array[i]=array[i+1];
            flag=1;
        }
    }
    *size=*size-1;
}

/***Calculate Lengths and Send them to the corresponding Node***/
void sendLengths(int size,int noProcesses)
{
    int i,partLength;
    if(size%noProcesses!=0)
    {
        int left=size-(size/noProcesses)*noProcesses;  //Split the size in as close to equal as possible parts
        partLength=(size/noProcesses)+1;
        for(i=1; i<left; i++)    //start from 1 because we create the zero one through the main function
            MPI_Send(&partLength,1,MPI_INT,i,1,MPI_COMM_WORLD);
        partLength-=1;
        for(i=left; i<noProcesses; i++)
            MPI_Send(&partLength,1,MPI_INT,i,1,MPI_COMM_WORLD);
    }
    else
    {
        partLength=size/noProcesses;
        for(i=1; i<noProcesses; i++)
            MPI_Send(&partLength,1,MPI_INT,i,1,MPI_COMM_WORLD);
    }
}

/****Swaps two values in an array****/
void swap_values(int *array,int x,int y)
{
    int temp;
    temp=array[x];
    array[x]=array[y];
    array[y]=temp;
}

/*****Send random numbers to every node.*****/
void generateNumbers(int *numberPart,int partLength, int cal)
{
    srand((cal+1)*time(NULL));     //Generate number to fill the array
    int i;
    for(i=0; i<partLength; i++)
        numberPart[i]=rand()-rand();
}

/***Validates the stability of the operation****/
void validation(int noProcesses,int median,int partLength,int size,int *numberPart,int processId,int master)
{
    MPI_Bcast(&median,1,MPI_INT,0,sub_comm);
    int countMin=0;
    int countMax=0;
    int countEq=0;
    int sumMax=0,sumMin=0,sumEq=0,i;

    int *recvBuff = (int*)malloc(size*sizeof(int));;

    for(i=0; i<partLength; i++)
    {
        if(numberPart[i]>median)
        {
            countMax++;

        }
        else if(numberPart[i]<median)
        {
            countMin++;
        }
        else
        {
            countEq++;
        }

    }

    MPI_Reduce(&countMax,&sumMax,1,MPI_INT,MPI_SUM,master,sub_comm);
    MPI_Reduce(&countMin,&sumMin,1,MPI_INT,MPI_SUM,master,sub_comm);
    MPI_Reduce(&countEq,&sumEq,1,MPI_INT,MPI_SUM,master,sub_comm);
    if(processId==0)
    {
        if((sumMax<=size/2)&&(sumMin<=size/2))   //Checks if both the lower and higher values occupy less than 50% of the total array.
        {
            printf("VALIDATION PASSED! master : %d procId : %d \n",master,processId);

        }
        else
            printf("VALIDATION FAILED!\n");


        printf("Values greater than median: %d\n",sumMax);
        printf("Values equal to median: %d\n",sumEq);
        printf("Values lower than median: %d\n\n\n",sumMin);
    }
    /// ITS TIME TO SNIFF THE VALUES THAT MASTER IS SENDING AND STORE THEM INTO MY NUMBERPART ARRAY

    /** -----------------------------------------------------------**/
    if(noProcesses>1)
    {
        if(processId == 0)
        {

            /// add to recv buff my local infos
            for(int i=0; i<partLength; i++)
            {
                recvBuff[i]=numberPart[i];
            }
        }
        MPI_Gather(numberPart,partLength,MPI_INT,recvBuff,partLength,MPI_INT,0,sub_comm);
        partitionAndSendArrays(processId,noProcesses,sumMax,sumEq,sumMin,median,partLength,size,recvBuff,numberPart);
    }

}

/***Validates the stability of the operation (Single Threaded)****/
void validationST(int median,int size,int *numberPart)
{
    int countMin=0;
    int countMax=0;
    int countEq=0;
    int i;
    for(i=0; i<size; i++)
    {
        if(numberPart[i]>median)
            countMax++;
        else if(numberPart[i]<median)
            countMin++;
        else
            countEq++;
    }
    if((countMax<=size/2)&&(countMin<=size/2))  //Checks if both the lower and higher values occupy less than 50% of the total array.
        printf("VALIDATION PASSED!\n");
    else
        printf("VALIDATION FAILED!\n");

    printf("Values greater than median: %d\n",countMax);
    printf("Values equal to median: %d\n",countEq);
    printf("Values lower than median: %d\n",countMin);
}

/****Part executed only by the Master Node****/
int masterPart(int noProcesses,int processId,int size,int partLength,int *numberPart,int masterId) //MASTER NODE CODE
{
    int elements,i,keepBigSet,sumSets,finalize,median,randomNode,pivot,k,tempPivot;
    int endSmall=0;
    int dropoutFlag=0;
    int endBig=0;
    int *arraySmall,*arrayBig,*arrayToUse,*activeNodes;
    int activeSize=noProcesses;
    int stillActive=1;
    int oldSumSets=-1;
    int checkIdentical=0;
    int useNewPivot=0;
    int *pivotArray;
    k=(int)size/2+1; //It is done so in order to find the right median in an even numbered array.
    elements=partLength;
    activeNodes=(int *)malloc(noProcesses*sizeof(int));  //we create the array that contains the active nodes.
    arrayToUse=numberPart;
    pivotArray=(int*)malloc(noProcesses*sizeof(int));  //Used for special occasions to gather values different than the pivot.
    int master = 0;
    for(i=0; i<activeSize; i++)
    {
        activeNodes[i]=i;
    }
    int randomCounter=0;
    int randomCounter2=0;
    struct timeval first, second, lapsed;
    struct timezone tzp;
    gettimeofday(&first, &tzp);
    for(;;)   //Begin the infinite loop until the median is found.
    {
        int counter=0;
        useNewPivot=0;
        if(stillActive==1&&checkIdentical!=0)  //If i still have values in my array and the Sumed Big Set is identical to the previous one, check for identical values.
        {
            for(i=0; i<elements; i++)
            {
                if(pivot==arrayToUse[i])
                    counter++;
                else
                {
                    useNewPivot=1;
                    tempPivot=arrayToUse[i];
                    break;
                }
            }
        }
        if(checkIdentical!=0)
        {
            int useNewPivotMax=0;
            MPI_Reduce(&useNewPivot,&useNewPivotMax,1,MPI_INT,MPI_MAX,master,sub_comm); //FIRST(OPTIONAL) REDUCE : MAX useNewPivot
            if(useNewPivotMax!=1)    //That means that the only values left are equal to the pivot!
            {
                median=pivot;
                finalize=1;
                MPI_Bcast(&finalize,1,MPI_INT,master,sub_comm); //FIRST(OPTIONAL) BROADCAST : WAIT FOR FINALIZE COMMAND OR NOT
                gettimeofday(&second, &tzp);
                if(first.tv_usec>second.tv_usec)
                {
                    second.tv_usec += 1000000;
                    second.tv_sec--;
                }
                lapsed.tv_usec = second.tv_usec - first.tv_usec;
                lapsed.tv_sec = second.tv_sec - first.tv_sec;
                printf("Time elapsed: %lu, %lu s\n", lapsed.tv_sec, lapsed.tv_usec);
                validation(noProcesses,median,partLength,size,numberPart,processId,master);
                //MPI_Finalize();
                free(pivotArray);
                return median;
            }
            else
            {
                finalize=0;
                int useit=0;
                randomCounter2++;
                MPI_Bcast(&finalize,1,MPI_INT,master,sub_comm);
                MPI_Gather(&useNewPivot, 1, MPI_INT, pivotArray, 1, MPI_INT, master,sub_comm); //Gather every value and chose a node to change the pivot.
                for(i=0; i<activeSize; i++)
                {
                    if(pivotArray[i]==1)
                    {
                        if((randomCounter2>1)&&(randomNode!=activeNodes[i]))  //Check if the same node has already been used in a similar operation.
                        {
                            useit=1;
                            randomNode=activeNodes[i];
                            randomCounter2=0;
                            break;
                        }
                        else if(randomCounter2<2)
                        {
                            useit=1;
                            randomNode=activeNodes[i];
                            break;
                        }
                    }
                }
                if(useit!=0)
                    useNewPivot=1;
                else
                    useNewPivot=0;
            }
        }
        if(useNewPivot!=0)
            MPI_Bcast(&randomNode,1,MPI_INT,master,sub_comm);  //THIRD(OPTIONAL) BROADCAST : BROADCAST THE SPECIAL NODE
        if(useNewPivot==0)  //if we didnt choose a special Node, choose the node that will pick the pivot in a clockwise manner. Only selects one of the active nodes.
        {
            if(randomCounter>=activeSize)
                randomCounter=0; //Fail safe
            randomNode=activeNodes[randomCounter];
            randomCounter++;			//Increase the counter
            MPI_Bcast(&randomNode,1,MPI_INT,master,sub_comm);   //FIRST BROADCAST : SENDING randomnode, who will chose
        }
        if(randomNode==processId)  //If i am to choose the pivot.....
        {
            if(useNewPivot==0)
            {
                srand(time(NULL));
                pivot=arrayToUse[rand() % elements];
                MPI_Bcast(&pivot,1,MPI_INT,master,sub_comm); //SECOND BROADCAST : SENDING PIVOT   k ton stelnw sto lao
            }
            else
            {
                MPI_Bcast(&tempPivot,1,MPI_INT,master,sub_comm); //SECOND BROADCAST : SENDING PIVOT   k ton stelnw sto lao
                pivot=tempPivot;
            }
        }
        else //If not.. wait for the pivot to be received.
            MPI_Bcast(&pivot,1,MPI_INT,randomNode,sub_comm);  // SECOND BROADCAST : RECEIVING PIVOT
        if(stillActive==1)  //If i still have values in my array.. proceed
        {
            partition(arrayToUse,elements,pivot,&arraySmall,&arrayBig,&endSmall,&endBig);  //I partition my array  // endsmall=number of elements in small array, it may be 0
            // endbig=number of elements in big array, it may be 0
            //arraysmall = Points to the position of the small array.NULL if the array is empty
            //Same for arraybig
        }
        else  //If i'm not active endBig/endSmall has zero value.
        {
            endBig=0;
            endSmall=0;
        }
        sumSets=0;
        //We add the bigSet Values to decide if we keep the small or the big array
        MPI_Reduce(&endBig,&sumSets,1,MPI_INT,MPI_SUM,master,sub_comm);  //FIRST REDUCE : SUM OF BIG
        MPI_Bcast(&sumSets,1,MPI_INT,master,sub_comm);
        if(oldSumSets==sumSets)
            checkIdentical=1;
        else
        {
            oldSumSets=sumSets;
            checkIdentical=0;
        }
        //hmetabliti keepBigSet 0 h 1 einai boolean k me autin enimerwnw ton lao ti na kratisei to bigset h to smallset
        if(sumSets>k)   //an to sumofbigsets > k tote krataw to big SET
        {
            keepBigSet=1; //to dilwnw auto gt meta tha to steilw se olous
            if(endBig==0)
                dropoutFlag=1; //wraia.. edw an dw oti to bigset mou einai 0.. alla prepei na kratisw to bigset sikwnw auti ti simaia pou simainei tha ginw inactive ligo pio katw tha to deis
            else
            {
                arrayToUse=arrayBig; //thetw ton neo pinaka na einai o big
                elements=endBig; //thetw arithmo stoixeiwn iso me tou big
            }
        }
        else if(sumSets<k) //antistoixa an to sumofbigsets < k tote krataw to small set
        {
            keepBigSet=0;
            k=k-sumSets;
            if(endSmall==0)
                dropoutFlag=1; //antistoixa koitaw an tha ginw inactive..
            else
            {
                arrayToUse=arraySmall; //dinw times..
                elements=endSmall;
            }
        }
        else  //edw simainei k=sumofbigsetes ara briskw pivot k telos
        {
            median=pivot;
            finalize=1; //dilwnw finalaize =1
            MPI_Bcast(&finalize,1,MPI_INT,master,sub_comm); //to stelnw se olous, oi opoioi an laboun finalize =1 tote kaloun MPI finalize k telos
            gettimeofday(&second, &tzp);
            if(first.tv_usec>second.tv_usec)
            {
                second.tv_usec += 1000000;
                second.tv_sec--;
            }
            lapsed.tv_usec = second.tv_usec - first.tv_usec;
            lapsed.tv_sec = second.tv_sec - first.tv_sec;
            printf("Time elapsed: %lu, %lu s\n", lapsed.tv_sec, lapsed.tv_usec);
            validation(noProcesses,median,partLength,size,numberPart,processId,master);
            //MPI_Finalize();
            free(pivotArray);
            return median;
        }
        finalize=0; //an den exw mpei sta if den exw steilei timi gia finalize.. oi alloi omws perimenoun na laboun kati, stelnw loipon to 0 pou simainei sunexizoume
        MPI_Bcast(&finalize,1,MPI_INT,master,sub_comm);	//SECOND BROADCAST : WAIT FOR FINALIZE COMMAND OR NOT
        //edw tous stelnw to keepbigset gia na doun ti tha dialeksoun
        MPI_Bcast(&keepBigSet,1,MPI_INT,master,sub_comm);    //THIRD BROADCAST: SEND keepBigset boolean
        if(dropoutFlag==1 && stillActive==1) //edw sumfwna me to dropoutflag pou orisame prin an einai 1 kalw tin sinartisi pou me petaei apo ton pinaka. episis koitaw na eimai active gt an me exei idi petaksei se proigoumeni epanalispi tote den xreiazetai na me ksanapetaksei
        {
            stillActive=0;
            removeElement(activeNodes, &activeSize,0);
        }
        int flag;
        //edw perimenw na akousw apo ton kathena an sunexizei active h oxi.. an oxi ton petaw.. an einai idi inactive apo prin stelnei kati allo (oxi 1)k den ton ksanapetaw
        for(i=0; i<activeSize; i++)
        {
            if(activeNodes[i]!=0)
            {
                MPI_Recv(&flag,1,MPI_INT,activeNodes[i],1,sub_comm,&Stat);  //FIRST RECEIVE : RECEIVE active or not
                if(flag==1)
                    removeElement(activeNodes, &activeSize, activeNodes[i]);
            }
        }
    }
    MPI_Comm_free(&sub_comm);
}

/***Executed only by Slave nodes!!*****/
void slavePart(int noProcess,int processId,int partLength,int *numberPart,int size,int colour)  //code here is for the cheap slaves :P
{
    int master = 0 ;
    int dropoutflag,elements,i,sumSets,finalize,keepBigSet,pivot,randomNode,tempPivot;
    int endSmall=0;
    int endBig=0;
    int *arraySmall,*arrayBig,*arrayToUse;
    arrayToUse=numberPart;
    elements=partLength;
    int stillActive=1;
    int *pivotArray;
    int oldSumSets=-1;
    int checkIdentical=0;
    int useNewPivot;
    for(;;)
    {
        finalize=0;
        int counter=0;
        useNewPivot=0;
        if(stillActive==1&&checkIdentical!=0)  //If i still have values in my array..   If the Sumed Big Set is identical to the previous one, check for identical values.
        {
            for(i=0; i<elements; i++)
            {
                if(pivot==arrayToUse[i])
                    counter++;
                else
                {
                    useNewPivot=1;
                    tempPivot=arrayToUse[i];
                    break;
                }
            }
        }
        if(checkIdentical!=0)
        {
            int useNewPivotMax=0;
            MPI_Reduce(&useNewPivot,&useNewPivotMax,1,MPI_INT,MPI_MAX,master,sub_comm);
            MPI_Bcast(&finalize,1,MPI_INT,master,sub_comm);//an o master apo to keepbigset k apo to count apofasisei oti teleiwsame mou stelnei 1, alliws 0 sunexizoume
            if(finalize==1)
            {
                int median=0;
                validation(noProcess,median,partLength,size,numberPart,processId,master);


                MPI_Finalize();
                return ;
            }
            else
            {
                MPI_Gather(&useNewPivot, 1, MPI_INT, pivotArray, 1, MPI_INT, master,sub_comm);
            }
        }
        MPI_Bcast(&randomNode,1,MPI_INT,master,sub_comm); //FIRST BROAD CAST : RECEIVING RANDOM NODE, perimenw na dw poios einaito done
        if(randomNode!=processId) //means I am not the one to chose pivot.. so I wait to receive the pivot
            MPI_Bcast(&pivot,1,MPI_INT,randomNode,sub_comm);	//SECOND BROADCAST : RECEIVING PIVOT
        else if(randomNode==processId) //I am choosing suckers
        {
            if(useNewPivot==0)
            {
                srand(time(NULL));
                pivot=arrayToUse[rand() % elements];
                MPI_Bcast(&pivot,1,MPI_INT,processId,sub_comm); //SECOND BROADCAST : SENDING PIVOT   k ton stelnw sto lao
            }
            else
            {
                MPI_Bcast(&tempPivot,1,MPI_INT,processId,sub_comm); //SECOND BROADCAST : SENDING PIVOT   k ton stelnw sto lao
                pivot=tempPivot;
            }
        }
        if(stillActive==1)   //an eksakolouthw na eimai active, trexw tin partition.. k to count kommati to opio eimape kapou exei problima
        {
            partition(arrayToUse,elements,pivot,&arraySmall,&arrayBig,&endSmall,&endBig);
        }
        else
        {
            endBig=0;
            endSmall=0;
        }
        //an eimai inactive stelnw endbig=0 gia to bigset pou den epireazei
        sumSets=0;
        MPI_Reduce(&endBig,&sumSets,1,MPI_INT,MPI_SUM,master,sub_comm); //FIRST REDUCE : SUM OF BIG, stelnw ola ta bigset gia na athroistoun sotn master
        MPI_Bcast(&sumSets,1,MPI_INT,master,sub_comm);
        if(oldSumSets==sumSets)
            checkIdentical=1;
        else
        {
            oldSumSets=sumSets;
            checkIdentical=0;
        }
        MPI_Bcast(&finalize,1,MPI_INT,0,sub_comm);//an o master apo to keepbigset k apo to count apofasisei oti teleiwsame mou stelnei 1, alliws 0 sunexizoume
        if(finalize==1)
        {
            int median=0;///HERE LOOK
            validation(noProcess,median,partLength,size,numberPart,processId,master);

            //MPI_Finalize();
            return ;
        }
        MPI_Bcast(&keepBigSet,1,MPI_INT,0,sub_comm);//THIRD BROADCAST: Receive keepBigset boolean, edw lambanw an krataw to mikro i megalo set.
        //afou elaba ton keepbigset an eimai active krataw enan apo tous duo pinake small h big.. alliws den kanw tpt
        //edw antistoixa allazw tous pointers, k eksetazw an exw meinei xwris stoixeia tin opoia periptwsi sikwnw to dropoutflag k pio katw tha dilwsw na ginw inactive
        if(stillActive==1)
        {
            if(keepBigSet==1)
            {
                if(endBig==0)
                    dropoutflag=1;
                else
                {
                    arrayToUse=arrayBig;
                    elements=endBig;
                }
            }
            else if(keepBigSet==0)
            {
                if(endSmall==0)
                    dropoutflag=1;
                else
                {
                    arrayToUse=arraySmall;
                    elements=endSmall;
                }
            }
        }
        //edw einai ligo periploka grammeno, isws exei perita mesa alla, an eimai active k thelw na ginw inactive einai i prwti periptwsi, h deuteri einai eimai inactive hdh k i triti einai sunexizw dunamika
        if(dropoutflag==1 && stillActive==1)
        {
            MPI_Send(&dropoutflag,1,MPI_INT,master,1,sub_comm); //FIRST SEND : send active or not;
            stillActive=0;
        }
        else if(stillActive==0)
        {
            dropoutflag=-1;
            MPI_Send(&dropoutflag,1,MPI_INT,master,1,sub_comm); //FIRST SEND : send active or not;
        }
        else
        {
            dropoutflag=0;
            MPI_Send(&dropoutflag,1,MPI_INT,master,1,sub_comm); //FIRST SEND : send active or not;
        }
    }
    MPI_Comm_free(&sub_comm);
}


/*****MAIN!!!!!!!!!!*****/
int main (int argc, char **argv)
{
    int processId,noProcesses,size,partLength,median;
    int *numberPart;

    size=atoi(argv[1]);

    MPI_Init (&argc, &argv);	/* starts MPI */
    MPI_Comm_rank (MPI_COMM_WORLD, &processId);	/* get current process id */
    MPI_Comm_size (MPI_COMM_WORLD, &noProcesses);	/* get number of processes */

    /**FIRST ROUND! SEND LENGHTS AND GENERATE NUMBERS FOR ALL PROSECCES **/

    int* masterNodesArray;
    int level = 0,masterId ;
    float limit = log2(size);

    int colour ;
    int sub_size = size ;

    int*keepMedian = (int*)malloc(limit*sizeof(int));


    while(level < limit)
    {
        sub_size = size/pow(2,level);
        masterNodesArray = (int*)malloc(pow(2,level)*sizeof(int));
        masterId = 0;// Id = 0  will be always a master independentlly the partitions of the processes
        //printf("level : %d ----- process : %d \n",level,processId);
        masterNodesArray = selectMasters(level,noProcesses);
        //for(int index = 0;index<pow(2,level);index++){printf("masters :  %d \n",masterNodesArray[index]);}

        for(int index = 0 ; index<pow(2,level); index++)
        {
            if(processId == masterNodesArray[index])
            {
                masterId = processId;
                break;
            }
        }
        //left nearest master will be the colour of the sub-communicator

        if(masterId!=processId)
        {
            colour = nearestLeftMasterNode(masterNodesArray,processId,pow(2,level));
        }
        else
        {
            colour = masterId;
        }

        //printf("master-colour: %d \n ",colour);
        /**create a new sub-communicator based on colour**/

        MPI_Comm_split(MPI_COMM_WORLD,colour,processId,&sub_comm);
        int sub_Procsize = noProcesses/pow(2,level) ;
        int sub_ProcId ;

        //MPI_Comm_size(sub_comm,&sub_Procsize);
        MPI_Comm_rank(sub_comm,&sub_ProcId);

        printf("nOfProc. : %d colour: %d process : %d \n",sub_Procsize,colour,processId);

        /** SINGLE THREAD CODING IS HERE !!! ( AKA when level = log2(sz)) **/

        if(processId == 0 && level == 0)
        {
            printf("size: %d processes: %d \n\n",size,noProcesses);
            if(noProcesses>1)
            {
                if(size%noProcesses==0)
                    partLength=(size/noProcesses);
                else
                    partLength=(size/noProcesses)+1;
                sendLengths(size,noProcesses);
                numberPart=(int*)malloc(partLength*sizeof(int));
                generateNumbers(numberPart,partLength,processId);
            }
            else
            {
                numberPart=(int*)malloc(size*sizeof(int));
                generateNumbers(numberPart,size,processId);
                struct timeval first, second, lapsed;
                struct timezone tzp;
                gettimeofday(&first, &tzp);
                median=selection(numberPart,size);
                gettimeofday(&second, &tzp);
                if(first.tv_usec>second.tv_usec)
                {
                    second.tv_usec += 1000000;
                    second.tv_sec--;
                }
                lapsed.tv_usec = second.tv_usec - first.tv_usec;
                lapsed.tv_sec = second.tv_sec - first.tv_sec;
                validationST(median,size,numberPart);
                printf("Time elapsed: %lu, %lu s\n", lapsed.tv_sec, lapsed.tv_usec);
                printf("Median: %d\n",median);
                free(numberPart);
                //MPI_Finalize();
                return 0;
            }
        }
        else if(processId!=0 && level==0)
        {
            MPI_Recv(&partLength,1,MPI_INT,0,1,MPI_COMM_WORLD,&Stat);
            numberPart=(int*)malloc(partLength*sizeof(int));
            generateNumbers(numberPart,partLength,processId);
        }



        if(processId==masterId)
        {
            median=masterPart(sub_Procsize,sub_ProcId,sub_size,partLength,numberPart,0);
            keepMedian[level] = median;
            printf("Median: %d for level : %d master: %d   \n\n",median,level,masterId);
        }
        else
            slavePart(sub_Procsize,sub_ProcId,partLength,numberPart,sub_size,colour);


        level ++;
        free(masterNodesArray);
        //MPI_Comm_free(&sub_comm);
    }

    /** LEVEL == LOG2(SIZE).... AKA SINGLE THREAD VALIDATION !!!!**/
    struct timeval first, second, lapsed;
    struct timezone tzp;
    gettimeofday(&first, &tzp);
    median=selection(numberPart,partLength);
    gettimeofday(&second, &tzp);
    if(first.tv_usec>second.tv_usec)
    {
        second.tv_usec += 1000000;
        second.tv_sec--;
    }
    lapsed.tv_usec = second.tv_usec - first.tv_usec;
    lapsed.tv_sec = second.tv_sec - first.tv_sec;
    validationST(median,partLength,numberPart);
    printf("[+][+][+][+][+][+][+][+][+] SINGLE-THREAD HEREEE [+][+][+][+][+][+][+][+] \n\n");
    printf("Time elapsed: %lu, %lu s\n", lapsed.tv_sec, lapsed.tv_usec);
    printf("Median: %d\n",median);




    /** ITS TIME TO IMPLEMENT KNN ALGORITHM **/
    //let it be :
    int boundsOfk = 8; //aka 2^k
    printf("!!!!!!!!!!!!!!!!!!!!partlength [+] %d \n",partLength);
    allKnnAlgo(processId,numberPart,partLength,pow(2,boundsOfk),keepMedian,level);


    MPI_Finalize();

    return 0;
}

/*========================FIND MEDIAN FUNCTIONS====================================
 * ================================================================================
 * ================================================================================
*/


/****Partitions the Array into larger and smaller than the pivot values****/
void partition (int *array, int elements, int pivot, int **arraysmall, int **arraybig, int *endsmall, int *endbig)
{
    int right=elements-1;
    int left=0;
    int pos;
    if(elements==1)
    {
        if(pivot>array[0])
        {
            *endsmall=1;  //One value in the small part
            *endbig=0;   //Zero on the big one
            *arraysmall=array;   //There is no big array therefore NULL value
            *arraybig=NULL;
        }
        else if(pivot<=array[0])
        {
            *endsmall=0;    //The exact opposite of the above actions.
            *endbig=1;
            *arraysmall=NULL;
            *arraybig=array;
        }
    }
    else if(elements>1)
    {
        while(left<right)
        {
            while(array[left]<pivot)
            {
                left++;
                if(left>=elements)
                {
                    break;
                }
            }
            while(array[right]>=pivot)
            {
                right--;
                if(right<0)
                {
                    break;
                }
            }
            if(left<right)
            {
                swap_values(array,left,right);
            }
        }
        pos=right;
        if(pos<0)                   //Arrange the arrays so that they are split into two smaller ones.
        {
            //One containing the small ones. And one the big ones.
            *arraysmall=NULL;           //However these arrays are virtual meaning that we only save the pointer values of the beging and end
        }                               //of the "real" one.
        else
        {
            *arraysmall=array;
        }
        *endsmall=pos+1;
        *arraybig=&array[pos+1];
        *endbig=elements-pos-1;
    }
}


/***==============================================***/
/***==============================================***/
/***=============SERIAL SELECTION==============***/
/***==============================================***/
/***==============================================***/

int selection(int *array,int number)
{
    int *arraybig;
    int *arraysmall;
    int endsmall=0;
    int endbig=0;
    int *arraytobeused;
    int i;
    int counter=0;
    int k;
    int pivot;
    int median;
    k=(int)number/2+1;
    arraytobeused=array;
    for(;;)
    {
        pivot=arraytobeused[rand() % number];
        partition(arraytobeused,number,pivot,&arraysmall,&arraybig,&endsmall,&endbig);
        if(endbig>k)
        {
            number=endbig;
            arraytobeused=arraybig;
            for(i=0; i<endbig; i++)
            {
                if(pivot==arraybig[i])
                    counter++;
                else
                    break;
            }
            if(counter==endbig)
            {
                median=arraybig[0];
                break;
            }
            else
                counter=0;
            //end of count equals
        }
        else if(endbig<k)
        {
            number=endsmall;
            arraytobeused=arraysmall;
            k=k-endbig;
        }
        else
        {
            median=pivot;
            break;
        }
    }
    return median;
}

int* selectMasters(int level,int noProc)
{
    int *arrayToReturn = (int*)malloc(pow(2,level)*sizeof(int));
    for(int index = 0; index<pow(2,level); index++)
    {
        arrayToReturn[index] = index*noProc/pow(2,level);
    }
    return arrayToReturn;

}

int nearestLeftMasterNode(int* masterNodesArray,int processId,int masterArrayLen)
{
    int index = 0;
    while(processId > masterNodesArray[index] )
    {
        if(index == masterArrayLen-1)
        {
            return masterNodesArray[index];
        }
        index ++;

    }
    //if(processId == masterNodesArray[index]){index++;}
    return masterNodesArray[index-1];

}

void partitionAndSendArrays(int processId,int noProcesses,int sumMax,int sumEq,int sumMin,int median,int partLength,int size,int* recvBuff,int *numberPart)
{
    int* dataFromMaster;
    if(processId == 0)
    {
        printf(" DATA FROM GATHER-VALIDATION : \n");
        for(int i = 0; i<size; i++)
        {
            printf("received buffer : %d \n",recvBuff[i]);
        }
        printf("median : %d \n",median);







        int* lowerArray = (int*)malloc(sumMin+sumEq*sizeof(int)); ///+1 because of median !
        int* greaterArray = (int*)malloc(sumMax*sizeof(int));

        int index_low=0,index_great=0,index_eq=0;

        int hostOfLowProc ;
        hostOfLowProc =  (int)(sumMin/partLength);
        int hostOfMaxProc ;
        hostOfMaxProc = (int)(sumMax/partLength);
        int hostOfEqProc ;
        hostOfEqProc = noProcesses - hostOfLowProc - hostOfMaxProc ;

        int lowProc = 1;
        int eqProc = hostOfLowProc-1;
        int greatProc = hostOfLowProc+hostOfEqProc-1;

        int index=0;
        //MPI_Bcast(&median,1,MPI_INT,0,sub_comm); /** SEND MEDIAN TO SLAVES **/
        while(index<size)
        {
            if(recvBuff[index] < median)
            {
                lowerArray[index_low] = recvBuff[index];
                index_low++;
            }
            else if(recvBuff[index]==median)
            {
                lowerArray[sumMin+index_eq] = median ;
                index_eq++;
            }
            else
            {
                greaterArray[index_great]=recvBuff[index];
                index_great++;
            }
            index++;
        }

        int* sendArray = (int*)malloc(partLength*sizeof(int));
        int sendIndex = 0;
        int counter = 0;
        ///SENDLOWER AND EQUAL TO MEDIAN =========>

        for(counter = 0; counter<index_low+index_eq+1; counter++)
        {
            if(counter<partLength)
            {
                numberPart[counter]=lowerArray[counter];
            }
            else
            {
                sendArray[sendIndex] = lowerArray[counter];
                sendIndex++;
                if(sendIndex == partLength)
                {
                    MPI_Send(&sendArray,1,MPI_INT,lowProc,LOWER_TAG,sub_comm);
                    lowProc++;
                    sendIndex = 0;
                    free(sendArray);
                    sendArray = (int*)malloc(partLength*sizeof(int));
                }
            }
        }

        ///EXAGGERATION OPTION ;)
        counter = 0;
        while(sendIndex!=partLength)
        {
            sendArray[sendIndex] = greaterArray[counter];
            counter++;
            sendIndex++;
        }
        counter-- ;
        MPI_Send(&sendArray,1,MPI_INT,lowProc,LOWER_TAG,sub_comm);
        sendIndex = 0;
        free(sendArray);
        sendArray = (int*)malloc(partLength*sizeof(int));
        greatProc = lowProc+1;

        ///SEND CLEAN-GREATER =========>
        while(counter<index_great-1)///!!
        {
            sendArray[sendIndex] = greaterArray[counter];
            sendIndex++;
            if(sendIndex == partLength)
            {
                MPI_Send(&sendArray,1,MPI_INT,greatProc,HIGHER_TAG,sub_comm);
                greatProc++;
                sendIndex = 0;
                free(sendArray);
                sendArray = (int*)malloc(partLength*sizeof(int));
            }
            counter++;
        }

    }
    else
    {

        dataFromMaster = (int*)malloc(partLength*sizeof(int));
        MPI_Recv(&dataFromMaster,1,MPI_INT,0,MPI_ANY_TAG,sub_comm,&Stat);
        numberPart = dataFromMaster ;

    }
}


void allKnnAlgo(int processId,int* numberPart,int partLength,int boundsOfk,int* keepMedian,int level)
{
    int stdPoint,direction,brother; // ==steadyPoint

    /** IMPORTANT DECISSION ... WHO AM I ? RIGHT SON ? OR LEFT SON? HMMMM.... **/
    if(processId == 0)
    {
        direction = LEFT ;
        brother = 1;
    }
    else
    {
        if(processId%2 == 0)
        {
            direction = LEFT ;
            brother = processId+1;
        }
        else
        {
            direction = RIGHT ;
            brother = processId-1;
        }
    }


    printf("!!!!!!!!!!!! me : %d my    bro : %d   direction : %d  !!!!!!!!!!!!!!!!!!!!!!!\n",processId,brother,direction);




    for(int index = 0; index<partLength; index++)
    {
        stdPoint = numberPart[index];
        findNN(brother,processId,numberPart,partLength,keepMedian,2,stdPoint,direction,level);
    }

}


void findNN(int brother,int processId,int*numberPart,int partLength,int*keepMedian,int boundsOfk,int stdPoint,int direction,int level)
{
    int* localNN =(int*)malloc(boundsOfk*sizeof(int));
    int *recvBuff = (int*)malloc(boundsOfk*sizeof(int));
    int median = keepMedian[level],keepIndex ;
    int previousDist = 0 ;
    printf("here ? \n");
    for(int index = 0; index < boundsOfk; index++){
        keepIndex = findMinDistance(numberPart,partLength,previousDist,stdPoint,median);
        previousDist = abs(numberPart[keepIndex] - stdPoint);
        localNN[index] = numberPart[keepIndex];
    }
///NOW I HAVE THE LOCAL INFOS THAT I NEED
    int cond ;

    printf("here2? \n");


    if(direction == RIGHT){

        if(previousDist> median){
            cond = 0 ;
            MPI_Send(&cond,1,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD);
            MPI_Send(&stdPoint,1,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD);
            MPI_Send(&localNN,boundsOfk,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD);
            MPI_Recv(&recvBuff,boundsOfk,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD,&Stat);
            localNN = recvBuff ;
        }
        else{cond = -1; MPI_Send(&cond,1,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD);}
    }
    else if(direction == LEFT){
        MPI_Recv(&cond,1,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD,&Stat);
        if(cond == 0){
            int point;
            MPI_Recv(&point,1,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD,&Stat);

            MPI_Recv(&recvBuff,boundsOfk,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD,&Stat);

            int tryDist ;
            int minDist ;

            for(int index1 = 0;index1< boundsOfk;index1++){
                minDist = abs(recvBuff[index1] - point);
                for(int index2 = 0 ;index2<partLength;index2++){
                    tryDist = abs(numberPart[index2] - point);
                    if(tryDist<minDist){
                        minDist = tryDist;
                        recvBuff[index1] = numberPart[index2];
                    }
                }
            }


            MPI_Send(&recvBuff,boundsOfk,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD);
        }
    }





    if(direction == LEFT){

        if(previousDist> median){
            cond = 0 ;
            MPI_Send(&cond,1,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD);
            MPI_Send(&stdPoint,1,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD);
            MPI_Send(&localNN,boundsOfk,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD);

            MPI_Recv(&recvBuff,boundsOfk,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD,&Stat);
            localNN = recvBuff ;
        }
        else{cond = -1; MPI_Send(&cond,1,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD);}
    }
    else if(direction == RIGHT){
        MPI_Recv(&cond,1,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD,&Stat);
        if(cond == 0){
            int point;
            MPI_Recv(&point,1,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD,&Stat);
            MPI_Recv(&recvBuff,boundsOfk,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD,&Stat);
            int tryDist ;
            int minDist ;

            for(int index1 = 0;index1< boundsOfk;index1++){
                minDist = abs(recvBuff[index1] - point);
                for(int index2 = 0 ;index2<partLength;index2++){
                    tryDist = abs(numberPart[index2] - point);
                    if(tryDist<minDist){
                        minDist = tryDist;
                        recvBuff[index1] = numberPart[index2];
                    }
                }
            }


            MPI_Send(&recvBuff,boundsOfk,MPI_INT,brother,KNN_TAG,MPI_COMM_WORLD);
        }
    }

}


int findMinDistance(int *numberPart,int partLength,int previousDist,int stdPoint, int median){

    int min = median ;
    int tryDist,keepIndex ;

    for(int counter = 0; counter<partLength;counter++){
        tryDist = abs(numberPart[counter]-stdPoint);
        if(tryDist<min && tryDist>previousDist){
            keepIndex = counter ;
        }
    }

    return keepIndex;

}


