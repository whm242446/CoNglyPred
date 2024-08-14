# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/inspur/whm409100220068/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/inspur/whm409100220068/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/inspur/whm409100220068/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/inspur/whm409100220068/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

export PATH="/home/inspur/whm409100220068/anaconda3/bin:$PATH"
export PATH=/home/inspur/whm409100220068/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/home/inspur/whm409100220068/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/home/inspur/whm409100220068/cuda-11.8
export LD_LIBRARY_PATH=/home/inspur/whm409100220068/anaconda3/envs/esm/lib:$LD_LIBRARY_PATH

#setting environment for gcc-11.1.0
export PATH=/home/software/gcc/gcc-11.1.0/bin:$PATH
export LD_LIBRARY_PATH=/home/software/gcc/gmp-6.2.1/lib:/home/software/gcc/mpfr-4.1.0/lib:/home/software/gcc/mpc-1.2.1/lib:/home/software/gcc/gcc-11.1.0/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/software/gcc/gmp-6.2.1/lib:/home/software/gcc/mpfr-4.1.0/lib:/home/software/gcc/mpc-1.2.1/lib:/home/software/gcc/gcc-11.1.0/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=/home/software/gcc/gcc-11.1.0/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=home/software/gcc/gcc-11.1.0/include:$CPLUS_INCLUDE_PATH


