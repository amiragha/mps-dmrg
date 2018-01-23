abstract type Measurement end
abstract type ExpectationMPS <: Measurement end

mutable struct CorrMPS{T<:Number} <: ExpectationMPS
    ops::Vector{T}
    locations_list::Vector{Vector{Int64}}
    mps_length::Int64
    values::Vector{T}

    function CorrMPS{T}(ops::Vector{T},
                        locations_list::Vector{Vector{T}},
                        mps_length::Int64) where {T<:Number}

        values = zeros(T, length(locations_list))
        return new(ops, locations_list, mps_length, length, values)
    end
end

mutable struct MPOExpectation{T<:Number} <: ExpectationMPS
    mpo::MPO{T}
    mps_length::Int64
    value::Vector{T}

    function CorrMPS{T}(ops::Vector{T},
                        locations_list::Vector{Vector{T}},
                        mps_length::Int64) where {T<:Number}

        values = zeros(T, length(locations_list))
        return new(ops, locations_list, mps_length, length, values)
    end
end
