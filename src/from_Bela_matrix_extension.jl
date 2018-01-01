# this is Bela's code translated to Julia for comparison
function extend_M(M, s)
    Mp = zeros(2,2,2,2,2,2,2,2)
    M0 = reshape(M, 2,2,2,2)

    if s == 0
        for index in CartesianRange(size(Mp))
            ol0,ol1,or0,or1,il0,il1,ir0,ir1 = index.I
            if ol1==il1 && or1==ir1
                Mp[index] = M0[ol0,or0,il0,ir0]
            end
        end
    elseif s == 1
        for index in CartesianRange(size(Mp))
            ol0,ol1,or0,or1,il0,il1,ir0,ir1 = index.I
            if ol0==il0 && or0==ir0
                Mp[index] = M0[ol1,or1,il1,ir1]
            end
        end
    end
    return reshape(Mp, 16, 16)
end

function extend_M_sign(M, s)
    Mp = zeros(2,2,2,2,2,2,2,2)
    M0 = reshape(M, 2,2,2,2)

    if s == 0
        for index in CartesianRange(size(Mp))
            ol0,ol1,or0,or1,il0,il1,ir0,ir1 = index.I
            if ol1==il1 && or1==ir1
                if ol0 != il0 && il1 == 2
                    sign = -1
                else
                    sign = +1
                end
                Mp[index] = sign*M0[ol0,or0,il0,ir0]
            end
         end
    elseif s == 1
        for index in CartesianRange(size(Mp))
            ol0,ol1,or0,or1,il0,il1,ir0,ir1 = index.I
            if ol0==il0 && or0==ir0
                if ol1 != il1 && ir0 == 2
                    sign = -1
                else
                    sign = +1
                end
                Mp[index] = sign*M0[ol1,or1,il1,ir1]
            end
        end
    end
    return reshape(Mp, 16, 16)
end

function double_U(ugate; applysigns=true)
    if applysigns
        return extend_M_sign(ugate, 0) * extend_M_sign(ugate, 1)
    else
        return extend_M(ugate, 0) * extend_M(ugate, 1)
    end
end
