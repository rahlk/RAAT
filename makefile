typo:   
	@- git status
	@- git add --all .
	@- git commit -am "commit with a makefile"
	@- git push origin master # <== update as needed

commit: 
	@- git status
	@- git commit -a
	@- git push origin master

update: 
	@- git pull origin master

status: 
	@- git status