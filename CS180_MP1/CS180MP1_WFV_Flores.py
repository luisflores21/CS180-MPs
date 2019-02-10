import heapq
import sys

#Read given from file
filer = sys.argv[1]
fp = open(filer, "r")

#Determine the dimensions of the given
lines = fp.read().splitlines()
checkdim = lines[0].split(" ")
truedim = len(checkdim)

#Determine which heuristic to use
def chooseH(chooser, curr, goal, glist, zind):
    h = 0
    if chooser == 1:
        #Misplaced Tiles
        dex = 0
        for x in goal:
            if curr[dex] == x and x != "0":
                h+=1
    elif chooser == 2:
        #Linear Conflict
        for x in range(truedim):
            for y in range(truedim):
                tk = curr[x][y]
                #remember that tj is always to the right of tk by definition
                ay = y
                if tk == '0':
                    continue
                while ay < truedim:
                    #this is for looking for goal state
                    if goal[x][ay] == tk:
                        gok = ay
                        break
                    ay += 1
                    #print ay
                if ay >= truedim:
                    #this means goal state is not here
                    continue
                ay = y+1
                while ay < truedim:
                    #this part searches for tj
                    tj = curr[x][ay]
                    aj = y
                    ay +=1
                    if tj == '0':
                        continue
                    while aj < truedim:
                        #this is for looking for goal state of tj
                        if goal[x][aj] == tj:
                            goj = aj
                            break
                        aj += 1
                    if aj >= truedim:
                        #this means goal state is not here
                        continue
                    #making it this far, it means both goal states are in the same line
                    if goj < gok:
                        #if goal of tj is to the left of goal of tk
                        h += 2
        #checks vertically
        for x in range(truedim):
            for y in range(truedim):
                tk = curr[y][x]
                #remember that tj is always below tk by definition
                ay = x
                if tk == '0':
                    continue
                while ay < truedim:
                    #this is for looking for goal state
                    if goal[ay][y] == tk:
                        gok = ay
                        break
                    ay += 1
                    #print ay
                if ay >= truedim:
                    #this means goal state is not here
                    continue
                ay = x+1
                while ay < truedim:
                    #this part searches for tj
                    tj = curr[ay][y]
                    aj = x
                    ay +=1
                    if tj == '0':
                        continue
                    while aj < truedim:
                        #this is for looking for goal state of tj
                        if goal[aj][y] == tj:
                            goj = aj
                            break
                        aj += 1
                    if aj >= truedim:
                        #this means goal state is not here
                        continue
                    #making it this far, it means both goal states are in the same line
                    if goj < gok:
                        #if goal of tj is to the left of goal of tk
                        h += 2
    elif chooser == 3:
        #Tiles out of row and column
        rowout = 0
        colout = 0
        for x in range(truedim):
            for y in range(truedim):
                bamoo = -1
                bam = -1
                for z in range(truedim):
                    if curr[x][y] == goal[x][z] and (curr[x][y] != "0" and goal[x][z] != "0"):
                        bam = 1
                    if curr[x][y] == goal[z][y] and (curr[x][y] != "0" and goal[z][y] != "0"):
                        bamoo = 1
                    if bam == 1 and bamoo == 1:
                        break
                if bam == -1:
                    rowout +=1
                if bamoo == -1:
                    colout +=1
        #overshoots by 1 but idk why
        rowout -= 1
        colout -= 1
        h = rowout + colout
    elif chooser == 4:
        #Gaschnig's (N-maxswap)
        h = 0
        Plist = [0 for x in range(truedim*truedim)]
        Blist = [0 for x in range(truedim*truedim)]
        Glist = glist
        indz = (truedim*truedim)-1
        indp = 0
        indb = 0
        indz = zind
        if indz == 0:
            for x in range(truedim):
                for y in range(truedim):
                    Plist[indp] = int(curr[x][y])
                    Blist[Plist[indp]] = indb
                    indb += 1
                    indp += 1
            run = 0
            while(Plist != Glist):
                if Plist[indz] == 0:
                    for x in range(truedim*truedim):
                        if Plist[x] != Glist[x]:
                            gswap = Blist[Plist[x]]
                            Blist[Plist[x]] = Blist[Plist[indz]]
                            Blist[Plist[indz]] = gswap
                            gswap = Plist[x]
                            Plist[x] = Plist[indz]
                            Plist[indz] = gswap
                            break
                else:
                    n = 0
                    tempP = Plist[Blist[n]]
                    Plist[Blist[n]] = Plist[Blist[Blist[n]]]
                    Plist[Blist[Blist[n]]] = tempP
                    tempB = Blist[n]
                    Blist[n] = Blist[Blist[n]]
                    Blist[tempB] = tempB
                h += 1
        else:
            for x in range(truedim):
                for y in range(truedim):
                    Plist[indp] = int(curr[x][y])
                    if Plist[indp] == 0:
                        Plist[indp] = (truedim*truedim)-1
                    else:
                        Plist[indp] -= 1
                    Blist[Plist[indp]] = indb
                    indb += 1
                    indp += 1
            run = 0
            while(Plist != Glist):
                if Plist[indz] == (truedim*truedim)-1:
                    for x in range(truedim*truedim):
                        if Plist[x] != Glist[x]:
                            gswap = Blist[Plist[x]]
                            Blist[Plist[x]] = Blist[Plist[indz]]
                            Blist[Plist[indz]] = gswap
                            gswap = Plist[x]
                            Plist[x] = Plist[indz]
                            Plist[indz] = gswap
                            break
                else:
                    n = (truedim*truedim)-1
                    tempP = Plist[Blist[n]]
                    Plist[Blist[n]] = Plist[Blist[Blist[n]]]
                    Plist[Blist[Blist[n]]] = tempP
                    tempB = Blist[n]
                    Blist[n] = Blist[Blist[n]]
                    Blist[tempB] = tempB
                h += 1
    elif chooser == 5:
        #Manhattan distance
        h = 0
        xlist = [0 for x in range(truedim*truedim)]
        ylist = [0 for x in range(truedim*truedim)]

        for x in range(truedim):
            for y in range(truedim):
                ugh = int(goal[x][y])
                xlist[ugh] = x
                ylist[ugh] = y

        for x in range(truedim):
            for y in range(truedim):
                index = int(curr[x][y])
                if index == 0:
                    continue
                addx = abs(x - xlist[index])
                addy = abs(y - ylist[index])
                h = h + addx + addy
    else:
        h = 0
    return h

#Swap tiles and return a new board to be used to make BoardStates
def tileSwap(board, h, i, j, k):
    thetrueidee = ""
    tempstore = board[j][k]
    newboard = [[0 for x in range(truedim)] for y in range(truedim)]
    for xo in range(truedim):
        for ox in range(truedim):
            newboard[xo][ox] = board[xo][ox]
            if xo == j and ox == k:
                newboard[xo][ox] = '0'
            if xo == h and ox == i:
                newboard[xo][ox] = tempstore
            thetrueidee+=newboard[xo][ox]
            thetrueidee+=","
    return (newboard, thetrueidee)

#Check if goal state has been reached; return 1 if reached
def goalReached(board):
    tell = 0
    if board == goal:
        tell = 1
    return tell

#reconstruct path
def recPath(cameFrom, curr, moves,starting):
    total_path = []
    total_path.append(moves[curr])
    parcurr = cameFrom[curr]
    while parcurr != starting:
        total_path.append(moves[parcurr])
        parcurr = cameFrom[parcurr]
    total_path.append("start")
    total_path = total_path[::-1]
    print total_path

#Constructing the board using a 2D array
startboard = [[0 for x in range(truedim)] for y in range(truedim)]
stateidee = ""
for i in range(truedim):
    checkdim = lines[i].split(" ")
    for j in range(truedim):
        startboard[i][j] = checkdim[j]
        stateidee += startboard[i][j]
        stateidee += ","
        if startboard[i][j] == '0':
            startzr = i
            startzc = j

#Constructing goal state (purely just for checking)
glist = [0 for x in range(truedim*truedim)]
indg = 0
turner = 0
goal = [[0 for x in range(truedim)] for y in range(truedim)]
for i in range(truedim):
    checkdim = lines[i+truedim].split(" ")
    for j in range(truedim):
        goal[i][j] = checkdim[j]
        glist[indg] = int(goal[i][j])
        if indg == 0 and glist[indg] == 0:
            turner = 1
        if turner != 1:
            if glist[indg] == 0:
                glist[indg] = (truedim*truedim)-1
            else:
                glist[indg] -= 1
        indg+=1
if turner == 1:
    indz = 0
else:
    indz = (truedim*truedim)-1

#Initializing open and closed lists
#oplist is a heapq where the state's f and object are stored
#colist is a dictionary where keys are string forms of states
oplist = []
colist = {}
cameFrom = {}
gScore = {}
fScore = {}
moves = {}
achoo = int(input("Choose your HEUR HEUR!\n0 - No h\n1 - Misplaced Tiles\n2 - Linear Conflict\n3 - Out of row and column\n4 - NMaxSwap\n5 - Manhattan Distance\n"))
gScore[stateidee] = 0
fScore[stateidee] = gScore[stateidee] + chooseH(achoo,startboard,goal,glist,indz)
moves[stateidee] = "start"
heapq.heappush(oplist, (fScore[stateidee], startboard, stateidee))

#Actual algorithm
while oplist:
    #while the open list is not empty
    q = heapq.heappop(oplist)
    qstate = q[1]
    qid = ""
    for s in range(truedim):
        for w in range(truedim):
            qid += qstate[s][w]
            qid += ","
    if qid in colist:
        continue

    goalee = goalReached(qstate)
    if goalee == 1:
        recPath(cameFrom, qid, moves, stateidee)
        print "Number of configurations:"
        print len(colist)
        break

    #generate all possible moves from q (at most 4)
    for zoo in range(truedim):
        for ooz in range(truedim):
            if qstate[zoo][ooz] == '0':
                zrow = zoo
                zcol = ooz
                break
    possmoves = {}
    tempmoves = {}
    #add q to closed list
    colist[qid] = qstate

    #neighbor generation
    if zrow-1 >= 0:
        #swap with board[zrow-1][zcol]
        q1board, q1id = tileSwap(qstate, zrow, zcol, zrow-1, zcol)
        possmoves[q1id] = q1board
        tempmoves[q1id] = "down"
    if zrow+1 < truedim:
        #swap with board[zrow+1][zcol]
        q2board, q2id = tileSwap(qstate, zrow, zcol, zrow+1, zcol)
        possmoves[q2id] = q2board
        tempmoves[q2id] = "up"
    if zcol-1 >= 0:
        #swap with board[zrow][zcol-1]
        q3board, q3id = tileSwap(qstate, zrow, zcol, zrow, zcol-1)
        possmoves[q3id] = q3board
        tempmoves[q3id] = "right"
    if zcol+1 < truedim:
        #swap with board[zrow][zcol+1]
        q4board, q4id = tileSwap(qstate, zrow, zcol, zrow, zcol+1)
        possmoves[q4id] = q4board
        tempmoves[q4id] = "left"

    #iterate through all possible moves
    for x in possmoves:
        #Check if pattern is in the closed list
        if x in colist:
            continue
        #Check for better values of g per possmove
        tempG = gScore[qid]+1
        if x in gScore:
            if tempG >= gScore[x]:
                continue

        #Store best value for possmove
        cameFrom[x] = qid
        moves[x] = tempmoves[x]
        gScore[x] = tempG
        fScore[x] = gScore[x] + chooseH(achoo, possmoves[x], goal, glist, indz)

        #No need to check for presence of possmove in the open list
        #because it has been previously checked if a better possmove
        #exists, and if so, no state is pushed. Duplicate state will be
        #pushed if possmove exists but is better than the existing one
        #but the duplicate will be handled by code above
        heapq.heappush(oplist, (fScore[x], possmoves[x], x))

#Close file containing given
fp.close()
